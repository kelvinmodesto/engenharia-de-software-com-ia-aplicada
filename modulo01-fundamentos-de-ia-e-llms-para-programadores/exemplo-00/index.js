import tf from "@tensorflow/tfjs-node";

async function predict(model, person) {
  // transformar arrar para tensor
  const tfInput = tf.tensor2d(person);

  // faz a predição (saída será um vetor de 3 probabilidades)
  const pred = model.predict(tfInput);
  const predArray = await pred.array();
  return predArray[0].map((prob, index) => ({ prob, index }));
}

async function trainModel(inputXs, outputYs) {
  const model = tf.sequential();

  // Primeira camada da rede:
  // entrada de 7 posições (idade normalizada + 3 cores + localizacoes)
  //
  //
  // 80 neuronios: quanto mais neuronios, mais inteligente a rede pode ser
  // e demanda mais processameento
  //
  // A ReLU age como um filtro
  // Deixando somente os dados relevantes para seguirem viagem na rede
  // se for zero ou negativa pode descartar
  model.add(tf.layers.dense({ inputShape: [7], units: 80, activation: "relu" }));

  // saída: 3 neuronios
  // um para cada categoria
  model.add(tf.layers.dense({ units: 3, activation: "softmax" }));

  // compilando o modelo
  // optimizer Adam (Adaptive Moment Estimation)
  // é um treinador pessoal moderno para redes neurais:
  // ajusta os pesos de forma eficiente
  //
  // loss: categoricalCrossentropy
  // compara o modelo e deduz os scores de cada categoria
  // com a resposta certa
  // ex: a categoria premium será sempre [1,0,0]
  //
  // quanto mais distante da previsão do modelo da resposta correta, maior o erro
  // exemplo: classificação de imagens, recomendação, categorização de usuário
  model.compile({
    optimizer: "adam",
    loss: "categoricalCrossentropy",
    metrics: ["accuracy"],
  });
  // verbose: desabilita log interno
  // epoochs: qtd de vezes que vai rodar no dataset
  // shuffle: emabaralha dados para evitar viés
  await model.fit(inputXs, outputYs, {
    verbose: 0,
    epochs: 100,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, log) => console.log(`Epoch: ${epoch}: loss = ${log.loss}`),
    },
  });
  return model;
}

// Exemplo de pessoas para treino (cada pessoa com idade, cor e localização)
// const pessoas = [
//     { nome: "Erick", idade: 30, cor: "azul", localizacao: "São Paulo" },
//     { nome: "Ana", idade: 25, cor: "vermelho", localizacao: "Rio" },
//     { nome: "Carlos", idade: 40, cor: "verde", localizacao: "Curitiba" }
// ];

// Vetores de entrada com valores já normalizados e one-hot encoded
// Ordem: [idade_normalizada, azul, vermelho, verde, São Paulo, Rio, Curitiba]
// const tensorPessoas = [
//     [0.33, 1, 0, 0, 1, 0, 0], // Erick
//     [0, 0, 1, 0, 0, 1, 0],    // Ana
//     [1, 0, 0, 1, 0, 0, 1]     // Carlos
// ]

// Usamos apenas os dados numéricos, como a rede neural só entende números.
// tensorPessoasNormalizado corresponde ao dataset de entrada do modelo.
const tensorPessoasNormalizado = [
  [0.33, 1, 0, 0, 1, 0, 0], // Erick
  [0, 0, 1, 0, 0, 1, 0], // Ana
  [1, 0, 0, 1, 0, 0, 1], // Carlos
];

// Labels das categorias a serem previstas (one-hot encoded)
// [premium, medium, basic]
const labelsNomes = ["premium", "medium", "basic"]; // Ordem dos labels
const tensorLabels = [
  [1, 0, 0], // premium - Erick
  [0, 1, 0], // medium - Ana
  [0, 0, 1], // basic - Carlos
];

// Criamos tensores de entrada (xs) e saída (ys) para treinar o modelo
const inputXs = tf.tensor2d(tensorPessoasNormalizado);
const outputYs = tf.tensor2d(tensorLabels);

// quanto mais dado melhor
const model = await trainModel(inputXs, outputYs);

const person = { nome: "Zoro", idade: 28, cor: "verde", localizacao: "Curitiba" };

// Normalizando a idade da pessoa da prova real, usando o padrão do treino
// Exemplo: idade_min =  25, idade_max= 40 -> (28 - 25) / (40 - 15) = 0.2

const normalizedPerson = [
  [
    0.2,
    0, //azul
    0, //vermelho
    1, //verde
    0, // São Paulo
    0, // Rio
    1, // Curitiba
  ],
];

const predictions = await predict(model, normalizedPerson);
const results = predictions
  .sort((a, b) => b.prob - a.prob)
  .map((p) => `${labelsNomes[p.index]} (${(p.prob * 100).toFixed(2)})`)
  .join("\n");

console.log(results);
