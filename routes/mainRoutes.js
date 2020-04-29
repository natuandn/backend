const router = require("express").Router();
const fs = require("fs");
const path = require("path");
const tf = require("@tensorflow/tfjs");
const tfn = require("@tensorflow/tfjs-node");
const uuid = require("uuid/v4");
const {
  configModel,
  trainingData,
  trainModel,
  predictSample
} = require("../helpers/neural");
const { modelConfigValidation } = require("../helpers/formValidate");

var x_train;
var y_train;
var feature_count;
var output_count;

router.get("/", (req, res) => {
  res.send("WELCOME TO R-IOT SIMPLE TEACHABLE MACHINE");
});

//--------------------------------------------------
// NEURAL NETWORK ..................................
// config the layer and model
router.post("/neural/config_model", async (req, res, next) => {
  
  const { filename, featureHeader, outputHeader, outputCount } = req.body;
  try {
    // validate form
    await modelConfigValidation.validateAsync({
      filename,
      featureHeader,
      outputHeader,
      outputCount
    });

    
    const featureCount = featureHeader.length;
    // check if file is exist or not
    if (!fs.existsSync(path.join("./public/neural/uploads", filename))) {
      return res.status(400).json({
        fileError: `This file ${filename} was not exist`
      });
    }

    
    console.log("da vao day2!!");
    // create training data
    const { xs, ys, error } = await trainingData(
      filename,
      featureHeader,
      outputHeader,
      featureCount,
      outputCount
    );
    
    console.log("da vao day2!!!!");
    // if errors respone error to client and exit
    if (error) {
      return res.status(400).json({
        configErr: error
      });
    }

    
    console.log("da vao day2!!");
    // Set data to global variable x_train,y_train,feature_count and output count
    x_train = xs;
    y_train = ys;
    feature_count = Number(featureCount);
    output_count = Number(outputCount);
    // respone 200 code to client
    return res.json({
      status: "OKE",
      msg: "Config model finshed, ready for training"
    });
  } catch (e) {
    return res.status(400).send(e);
  }
});

router.post("/neural/train_model", async (req, res, next) => {
  if (!x_train || !y_train) {
    return res.status(400).send(`You need to config model in step 2 first`);
  }
  const { batchSize, epoch, learningRate } = req.body;

  //train model with mode for default we set 100 to number of unit node
  const model = configModel(feature_count, output_count, 100);
  const info = await trainModel(
    model,
    x_train,
    y_train,
    Number(epoch),
    Number.parseInt(batchSize),
    Number(learningRate)
  );
  console.log(`model info ${JSON.stringify(info)}`);

  // save mode to server
  let modelPath = uuid();
  await model.save(`file://./public/neural/models/${modelPath}`);
  //predict model
  const result = predictSample([30, 32, 0], model);
  console.log("result", result);
  // response to client
  res.json({
    status: "OK",
    msg: `model is trained`,
    data: {
      modelToken: modelPath,
      trainingInfo: { ...info.history, ...{ epoch: info.epoch } }
    }
  });
});
router.get("/neural/test_model", async (req, res) => {
  const { model_token } = req.query;
  // create handler to get model and weight file in to handler variable
  const handler = tfn.io.fileSystem(
    `./public/neural/models/${model_token}/model.json`
  );
  // now we can use tf.loadLayerModel like when we using in browser
  const model = await tf.loadLayersModel(handler);
  const result = predictSample([30, 32, 0], model);

  res.send(result);
});

router.get("/neural/predict", async (req, res) => {
  const { input } = req.query;
  // create handler to get model and weight file in to handler variable
  const handler = tfn.io.fileSystem(`./public/neural/test_model/model.json`);
  // now we can use tf.loadLayerModel like when we using in browser
  const model = await tf.loadLayersModel(handler);
  const strInput = input.split(" ");
  const numInput = strInput.map(ele => {
    return parseInt(ele);
  });
  const result = predictSample(numInput, model);
  res.status(200).send(result);
});

//--------------------------------------------------
// END NEURAL NETWORK ..............................
//--------------------------------------------------

module.exports = router;
