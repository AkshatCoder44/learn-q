class simpleNN {
  constructor(inputCount) {
    this.inputCount = inputCount;
    this.layers = [];
    this.weights = [];
    this.biases = [];
    this.learningRate = 0.01;
  }

  // Add a layer
  add(size, activation = "linear") {
    this.layers.push({ size, activation });

    const prevSize =
      this.weights.length === 0
        ? this.inputCount
        : this.layers[this.layers.length - 2].size;

    const weightMatrix = Array.from({ length: size }, () =>
      Array.from({ length: prevSize }, () => Math.random() * 2 - 1)
    );

    const biasVector = Array.from({ length: size }, () => Math.random() * 2 - 1);

    this.weights.push(weightMatrix);
    this.biases.push(biasVector);
  }

  // Activation functions
  activate(x, type) {
    switch (type) {
      case "relu":
        return Math.max(0, x);
      case "sigmoid":
        return 1 / (1 + Math.exp(-x));
      case "tanh":
        return Math.tanh(x);
      default:
        return x;
    }
  }

  // Derivatives
  activateDerivative(x, type) {
    switch (type) {
      case "relu":
        return x > 0 ? 1 : 0;
      case "sigmoid":
        const s = 1 / (1 + Math.exp(-x));
        return s * (1 - s);
      case "tanh":
        return 1 - Math.tanh(x) ** 2;
      default:
        return 1;
    }
  }

  // Forward pass
  forward(input) {
    let activations = [input];
    let zs = [];

    for (let l = 0; l < this.layers.length; l++) {
      const w = this.weights[l];
      const b = this.biases[l];
      const actType = this.layers[l].activation;

      const prev = activations[l];
      let z = [];
      let a = [];

      for (let i = 0; i < w.length; i++) {
        let sum = b[i];
        for (let j = 0; j < w[i].length; j++) {
          sum += w[i][j] * prev[j];
        }
        z.push(sum);
        a.push(this.activate(sum, actType));
      }

      zs.push(z);
      activations.push(a);
    }

    return { activations, zs };
  }

  // Predict (supports single OR batch)
  predict(input) {
    if (!Array.isArray(input[0])) {
      return this.forward(input).activations.slice(-1)[0];
    } else {
      return input.map(i => this.predict(i));
    }
  }

  // Train (supports single OR batch)
  train(inputs, targets, epochs = 1000) {
    // Normalize to batch
    if (!Array.isArray(inputs[0])) inputs = [inputs];
    if (!Array.isArray(targets[0])) targets = [targets];

    for (let epoch = 0; epoch < epochs; epoch++) {
      for (let n = 0; n < inputs.length; n++) {
        const x = inputs[n];
        const y = targets[n];

        const { activations, zs } = this.forward(x);

        let deltas = new Array(this.layers.length);

        // Output layer
        const last = this.layers.length - 1;
        deltas[last] = [];

        for (let i = 0; i < this.layers[last].size; i++) {
          const error = activations[last + 1][i] - y[i];
          deltas[last][i] =
            error *
            this.activateDerivative(zs[last][i], this.layers[last].activation);
        }

        // Backprop
        for (let l = last - 1; l >= 0; l--) {
          deltas[l] = [];
          for (let i = 0; i < this.layers[l].size; i++) {
            let sum = 0;
            for (let j = 0; j < deltas[l + 1].length; j++) {
              sum += this.weights[l + 1][j][i] * deltas[l + 1][j];
            }
            deltas[l][i] =
              sum *
              this.activateDerivative(zs[l][i], this.layers[l].activation);
          }
        }

        // Update weights & biases
        for (let l = 0; l < this.weights.length; l++) {
          for (let i = 0; i < this.weights[l].length; i++) {
            for (let j = 0; j < this.weights[l][i].length; j++) {
              this.weights[l][i][j] -=
                this.learningRate * deltas[l][i] * activations[l][j];
            }
            this.biases[l][i] -= this.learningRate * deltas[l][i];
          }
        }
      }
    }
  }
}

const nn = new simpleNN(1);
nn.add(8, 'relu');
nn.add(3, 'linear');
function getState() {
  return 0.5; // later replace with snake info
}

function getNextState(state, action) {
  let change = (action - 1) * 0.1;
  return Math.min(1, Math.max(0, state + change));
}

function getReward(state, nextState) {
  return nextState - state;
}

let state = getState();
let gamma = 0.9;
let epsilon = 0.2;

for(let i =0; 100000>i; i++) {
  let qv = nn.predict([state]);
  let action;
  if(epsilon > Math.random()) {
    action = Math.floor(Math.random() * qv.length)
  } else {
    action = qv.indexOf(Math.max(...qv))
  }

  const nextState = getNextState(state, action);
  let reward = getReward(state, nextState);

  const nextv = nn.predict([nextState]);
  const maxq = Math.max(...nextv);

  const target = qv.slice();
  target[action] = reward + gamma * maxq;
  nn.train([[state]], [target], 1);

  if (state <= 0 || state >= 1) {
    state = getState();
  } else {
    state = nextState;
  }
  epsilon*=0.995;
}

for (let i = 0; i <= 1; i += 0.01) {
  let stateN = i;
  let qv = nn.predict([stateN]);
  let action = qv.indexOf(Math.max(...qv));

  console.log(
    `State: ${stateN.toFixed(2)} | Q: ${qv.map(v => v.toFixed(2))} | Best: ${action}`
  );
}
