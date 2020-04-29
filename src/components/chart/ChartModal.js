import React from "react";
import { Line } from "react-chartjs-2";
import { data } from "./chartConfig";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faWindowClose } from "@fortawesome/free-solid-svg-icons";

function ChartModal({ trainingInfo, isOpen, setIsOpenModal }) {
  React.useEffect(() => {
    data.labels = trainingInfo.epoch;
    data.datasets[0].data = trainingInfo.loss;
    data.datasets[1].data = trainingInfo.acc;
  }, [trainingInfo]);

  return (
    <div className={`modal-wrapper ${!isOpen ? "hidden" : ""}`}>
      <div className="row">
        <div className="col s8 offset-s2 ">
          <div className="model-head">
            <h5 className="white-text center">Training Info</h5>
            <button
              onClick={() => {
                setIsOpenModal(false);
              }}
              className="btn-floating btn-small waves-effect waves-light pink close-btn"
            >
              <FontAwesomeIcon icon={faWindowClose} />
            </button>
          </div>

          <Line data={data} />
        </div>
      </div>
    </div>
  );
}

export default ChartModal;
