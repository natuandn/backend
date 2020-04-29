const tf = require("@tensorflow/tfjs");
require("@tensorflow/tfjs-node");
const path = require("path");
const { normalize, readCSV } = require("./util");
const TRAIN_DATA_PATH = path.resolve("./public/neural/uploads");

/**
 * todo: setup training data
 * @param { String } file_name :String path to csv_file
 * @param { [ String ]} feature_header :The header of feature that used for training
 * @param {String or Number } output_header : Header csv that was the output user for training data
 */
async function trainingData(
  file_name,
  feature_header,
  output_header,
  feature_count,
  output_count
) {
  try {
    // read the csv file to get all data
    const csv_data = await readCSV(path.join(TRAIN_DATA_PATH, file_name));
    // checking if object does not including props with header csv
    if (isOwnProperty(feature_header, output_header, csv_data) === false) {
      return {
        xs: null,
        ys: null,
        error: `feature or output doesn't match with the header of csv file`
      };
    }
    // format csv data to json object that can used for training
    let csv_transform = csv_data.map(data => {
      const x = feature_header.map(label => Number(data[label]));
      return {
        xs: x,
        ys: data[output_header]
      };
    });

    const x_train = csv_transform.map(ele => ele.xs);
    const y_train = csv_transform.map(ele => ele.ys);

    const xs = tf.tensor2d(x_train, [x_train.length, Number(feature_count)]);
    // onehot encoding : [0,1,2,3] giving 1 => [0,1,0,0];
    const ys = tf.oneHot(tf.tensor1d(y_train, "int32"), Number(output_count));
    return {
      xs,
      ys,
      error: null
    };
  } catch (e) {
    return {
      xs: null,
      ys: null,
      error: e
    };
  }
}

// check props of object
function isOwnProperty(feature_header, output_header, arr) {
  let isValid = true;
  if (!arr[0].hasOwnProperty(output_header)) {
    isValid = false;
  }
  feature_header.map(label => {
    if (!arr[0].hasOwnProperty(label)) {
      isValid = false;
    }
  });

  return isValid;
}

/*
todo: setup models
**/

function configModel(feature_count, output_count, unit_count) {
  const model = tf.sequential();
  // hidden config
  model.add(
    tf.layers.dense({
      inputShape: [feature_count],
      activation: "sigmoid",
      units: unit_count
    })
  );
  // fully connected layer
  // model.add(tf.layers.dense({ units: 175, activation: "relu" }));
  //output config
  model.add(
    tf.layers.dense({
      inputShape: unit_count,
      units: output_count,
      activation: "softmax"
    })
  );
  return model;
}

/**
 * todo : Void function take the model and train it
 * @param {*} model
 * @param {*} xs
 * @param {*} ys
 */
async function trainModel(
  model,
  xs,
  ys,
  epochs = 100,
  batchSize = 32,
  learningRate = 0.001
) {
  model.summary();
  // compiling model
  model.compile({
    loss: tf.losses.meanSquaredError,
    optimizer: tf.train.adam(learningRate),
    metrics: ["acc"]
  });
  const info = await model.fit(xs, ys, {
    epochs: epochs,
    batchSize: batchSize
  });
  console.log(`train finish`);

  return info;
}
/*
todo: setup predict using model
**/
function predictSample(sample, model) {
  let result = model.predict(tf.tensor(sample, [1, sample.length])).arraySync();
  return result;
}

async function run() {
  const { xs, ys } = await trainingData(
    "IOT.csv",
    ["indoor", "outdoor", "someone_exist"],
    "fan",
    3,
    2
  );
  xs.print();
  ys.print();

  const model = configModel(3, 2, 100);
  const info = await trainModel(model, xs, ys);
  console.log(`model info ${JSON.stringify(info)}`);

  const result = predictSample([30, 32, 0], model);
  console.log("result");
  console.log(result);
}
// run();

module.exports = {
  configModel,
  trainingData,
  trainModel,
  predictSample
};
