import math

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as opt

class Encoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.embed = nn.Embedding(256, dim)

    def __call__(self, x: mx.array): return self.embed(x)

class Decoder(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.decode = nn.Linear(dim, 256)
        self.stop = nn.Linear(dim, 1)

    def __call__(self, x: mx.array): return self.decode(x), mx.sigmoid(self.stop(x))

class Layer(nn.Module):
    def __init__(self, dim: int, spread: int):
        super().__init__()

        halflives = mx.exp(mx.linspace(0.0, math.log(float(spread)), dim))
        retention = mx.exp(-math.log(2.0) / halflives)
        self.decay = mx.log(retention) - mx.log1p(-retention)
    
        self.states = mx.zeros((dim, ))
        self.decaytrace = mx.zeros((dim, ))
        self.embedtrace = mx.zeros((256, dim))
        
        self.norm = nn.LayerNorm(dim)
        self.weights = nn.Linear(dim, dim, bias = False)
        self.silu = nn.SiLU()

        self.freeze(keys = ['states', 'decaytrace', 'embedtrace'], recurse = False)        

    def __call__(self, enc: mx.array, x: mx.array, dummy: mx.array):
        decay = mx.sigmoid(self.decay)
        state = (decay * self.states) + enc + dummy

        return x + self.silu(self.weights(self.norm(state))), state, decay

class Model(nn.Module):
    def __init__(self, dim: int = 512, layers: int = 16, spread: int = 32, temp: float = 0.75, rate: float = 0.0005, bound: tuple[int, int] = (40000, 120000)):
        super().__init__()
        self.dim = dim
        self.layers = layers
        self.temp = temp

        self.encoder = Encoder(dim)
        self.decoder = Decoder(dim)

        self.blocks = [Layer(dim, spread) for _ in range(layers)]

        def lrfn(step: mx.array):
            progress = mx.clip((step.astype(mx.float32) + 1.0 - float(bound[0])) / float(bound[1] - bound[0]), 0.0, 1.0)
            return rate * (1.0 - 0.9 * progress)

        self.optimizer = opt.AdamW(learning_rate = lrfn)
        self.compiled = None

    def sample(self, output: mx.array, key: mx.array | None = None):
        probs = mx.softmax(output)
        entropy = -mx.sum(probs * mx.log(probs + 1e-8)) / mx.log(mx.array(256.0))

        temp = mx.maximum(0.1, self.temp * (1.0 - self.temp * entropy)).item()
        return mx.random.categorical(output / temp, key=key)

    def step(self, c: mx.array, dummies: mx.array | None = None, frozen: bool = False):
        if dummies is None: dummies = [mx.zeros((self.dim, )) for _ in range(self.layers)]

        enc = self.encoder(c)
        x = enc
            
        states, decays = [], []

        for i, layer in enumerate(self.blocks):
            x, state, decay = layer(enc, x, dummies[i])
            if frozen: layer.states = mx.stop_gradient(state)

            states.append(state)
            decays.append(decay)

        return (x, states, decays), self.decoder(x)

    def updategrads(self, grads):
        if self.compiled is None:
            self.optimizer.update(self, grads)
            mx.eval(self.parameters(), self.optimizer.state)
            state = [self.state, self.optimizer.state]

            def update(gradients):
                self.optimizer.update(self, gradients)
                return self.optimizer.state["step"]

            self.compiled = mx.compile(update, inputs = state, outputs = state)

        else:
            self.compiled(grads)
            mx.eval(self.parameters(), self.optimizer.state)

    def loss(self, x: mx.array, output: mx.array, stop: mx.array, nextb: int | None, end: bool, ce_only: bool = False,
             components: dict | None = None) -> mx.array:
        terms = {name: mx.array(0.0) for name in ('variance', 'latent_prediction', 'cross_entropy', 'stop')}
        if not ce_only:
            loss = mx.maximum(0.0, 1.0 - mx.sqrt(mx.var(x) + 1e-4))
            terms['variance'] = loss
            if nextb is not None:
                n = mx.array(nextb)
                tgt = mx.stop_gradient(self.encoder(n))

                terms['latent_prediction'] = mx.mean(mx.square(x - tgt))
                terms['cross_entropy'] = -output[n] + mx.logsumexp(output)
                terms['stop'] = mx.mean(mx.square(stop - mx.array([1.0 if end else 0.0])))
                loss = loss + terms['latent_prediction']
                loss = loss - output[n] + mx.logsumexp(output)
                loss = loss + terms['stop']
        else:
            if nextb is not None: loss = -output[mx.array(nextb)] + mx.logsumexp(output)
            else: loss = 0.0
            terms['cross_entropy'] = mx.array(loss)

        if components is not None:
            components.update(terms)

        return loss

    def train_step(self, currb: int, nextb: int | None, end: bool, ce_only: bool = False,
                   metrics: dict | None = None) -> tuple[mx.array, mx.array, mx.array]:
        c = mx.array(currb)
        p = self.trainable_parameters()

        def fwd(params, dummies: list[mx.array]):
            self.update(params)
            (x, states, decays), (output, stop) = self.step(c, dummies)
            components = {}
            loss = self.loss(x, output, stop, nextb, end, ce_only, components)
            return loss, (states, decays, output, stop, components)

        (loss, (states, decays, output, stop, components)), (grads, dlds_s) = mx.value_and_grad(
            fwd, argnums = (0, 1)
        )(p, [mx.zeros((self.dim, )) for _ in range(self.layers)])

        self.update(p)

        c_range = (mx.arange(256) == c)[:, None].astype(mx.float32)
        for i, layer in enumerate(self.blocks):
            dlds = dlds_s[i]

            decay_embedtrace = layer.embedtrace * decays[i]
            embedtrace = decay_embedtrace + c_range
            grads["encoder"]["embed"]["weight"] += dlds * decay_embedtrace

            decaytrace = (decays[i] * layer.decaytrace) + (decays[i] * (1.0 - decays[i]) * layer.states)
            grads["blocks"][i]["decay"] = dlds * decaytrace

            layer.states = mx.stop_gradient(states[i])
            layer.decaytrace = mx.stop_gradient(decaytrace)
            layer.embedtrace = mx.stop_gradient(embedtrace)

        mx.eval(*[
            value
            for layer in self.blocks
            for value in (layer.states, layer.decaytrace, layer.embedtrace)
        ])

        if metrics is not None:
            mx.eval(loss, *components.values())
            metrics.update({name: float(value.item()) for name, value in components.items()})
        self.updategrads(grads)
        return output, stop, loss

    def __call__(self, currb: int, nextb: int | None, end: bool, frozen: bool, ce_only: bool):
        c = mx.array(currb)

        if frozen:
            _, (output, stop) = self.step(c, frozen = True)

            mx.eval(*[layer.states for layer in self.blocks])
            return self.sample(output).item(), stop.item()

        output, stop, _ = self.train_step(currb, nextb, end, ce_only)
        return self.sample(output).item(), stop.item()

    def reset(self):
        for layer in self.blocks:
            layer.states = mx.zeros((self.dim, ))

            layer.decaytrace = mx.zeros((self.dim, ))
            layer.embedtrace = mx.zeros((256, self.dim))

        mx.eval(*[layer.states for layer in self.blocks])

    def count(self) -> int:
        per_layer = self.dim * self.dim + 3 * self.dim
        return 256 * self.dim + self.layers * per_layer + 256 * self.dim + 256 + self.dim + 1
