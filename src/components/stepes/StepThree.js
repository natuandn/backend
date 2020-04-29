import React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faRunning } from "@fortawesome/free-solid-svg-icons";
import { modelDownLoad, trainModel } from "../../helpers/callApi";
import Loader from "../loader/Loader";
import ChartModal from "../chart/ChartModal";

function StepThree() {
  const [isTraining, setIsTraining] = React.useState(false);
  const [isOpenModal, setIsOpenModal] = React.useState(false);
  const [isTrainFinish, setIsTrainFinish] = React.useState(false);
  const [trainOption, setTrainOption] = React.useState({
    epoch: 20,
    learningRate: 0.001,
    batchSize: 32
  });
  const [trainingInfo, setTrainingInfo] = React.useState({
    loss: [],
    acc: [],
    epoch: []
  });
  const [modelToken, setModelToken] = React.useState("");
  const [errorMsg, setErrorMsg] = React.useState("");

  const handleSubmit = async e => {
    e.preventDefault();

    try {
      setIsTraining(true);
      setErrorMsg("");
      const res = await trainModel(trainOption);
      const { data } = res;
      setModelToken(data.data.modelToken);
      console.log(data.data.trainingInfo);

      setTrainingInfo(data.data.trainingInfo);
      setIsTraining(false);
      setIsTrainFinish(true);
    } catch (err) {
      setIsTraining(false);
      setErrorMsg(err.response && err.response.data);
    }
  };
  const handleOnChange = e => {
    setTrainOption({
      ...trainOption,
      [e.target.name]: e.target.value
    });
  };
  const exportModel = async () => {
    console.log(`export model`);
    await modelDownLoad(modelToken);
  };

  return (
    <div className="row">
      <div className="row ">
        <div className="col s12 m8 l8 offset-m2 offset-l2">
          <form onSubmit={handleSubmit} className="card z-depth-3">
            <div className="card-content">
              <span className="card-title pink-text center">
                Training option
              </span>
              <div className="row">
                <div className="input-field col s6">
                  <input
                    type="number"
                    name="epoch"
                    id="epoch"
                    onChange={handleOnChange}
                    value={trainOption.epoch}
                    autoFocus
                  />
                  <label htmlFor="epoch">Epoch</label>
                </div>
                <div className="input-field col s6">
                  <input
                    type="number"
                    step="0.001"
                    name="learningRate"
                    id="learningRate"
                    onChange={handleOnChange}
                    value={trainOption.learningRate}
                    autoFocus
                  />
                  <label htmlFor="learningRate">Learning Rate</label>
                </div>
                <div className="input-field col s12">
                  <input
                    type="number"
                    name="batchSize"
                    id="batchSize"
                    onChange={handleOnChange}
                    value={trainOption.batchSize}
                    autoFocus
                  />
                  <label htmlFor="batchSize">Batch Size</label>
                </div>
              </div>
            </div>
            <div className="card-action center">
              <button
                className="btn waves-effect waves-light center purple darken-2"
                name="action"
              >
                Train now
                <FontAwesomeIcon icon={faRunning} />
              </button>
            </div>
          </form>
        </div>
      </div>

      {isTrainFinish ? (
        <div className="row">
          <div className="col s12 l8 m8 offset-m2 offset-l2">
            <div className="card z-depth-3">
              <div className="card-content">
                <span className="card-title pink-text center">
                  Training Finish
                </span>

                <div className="pink-text center">
                  Your model is trained. You can check your model infomations or
                  export it right now !!
                  <div className="card-action center">
                    <button
                      className="btn waves-effect waves-light center purple darken-2 mr-20"
                      onClick={exportModel}
                    >
                      Export model
                    </button>
                    <button
                      className="btn waves-effect waves-light center purple darken-2"
                      onClick={() => {
                        setIsOpenModal(true);
                      }}
                    >
                      Check info
                    </button>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      ) : null}

      <ChartModal
        trainingInfo={trainingInfo}
        isOpen={isOpenModal}
        setIsOpenModal={setIsOpenModal}
      />
      {errorMsg ? <h5 className="red-text center ">{errorMsg}</h5> : null}
      {isTraining ? (
        <div className="center">
          <Loader />
          <h5 className="purple-text ">Training start...</h5>
        </div>
      ) : null}
    </div>
  );
}

export default StepThree;
