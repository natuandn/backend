import React from "react";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faArrowRight } from "@fortawesome/free-solid-svg-icons";
function StepTwoLoader() {
  return (
    <div className="col s12 loading-wrapper" action="">
      <div className="row">
        <div className=" col l6 s12 m6">
          <div className="loading-content "></div>
        </div>
        <div className="col l6 s12 m6">
          <div className="loading-content "></div>
        </div>
      </div>
      <div className="row">
        <div className=" col l6 s12 m6">
          <div className="loading-content "></div>
        </div>
        <div className="col l6 s12 m6">
          <div className="loading-content "></div>
        </div>
      </div>
      <h3 className="row center purple-text">Fetch new data from Server </h3>
    </div>
  );
}

export default StepTwoLoader;
