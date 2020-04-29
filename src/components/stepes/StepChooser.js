import React, {  } from "react";

function StepChooser({
  previous_step,
  next_step,
  currentStep,
  setCurrentStep
}) {
  let steps = ["Step 1", "Step 2", "Step 3"];
  return (
    <div className="row">
      <div className="col s12 purple darken-3 p-0 m-0">
        <ul className="tab">
          <li className="steper" onClick={() => previous_step()}>
            Prev
          </li>
          {steps.map((ele, index) => {
            if (index + 1 === currentStep) {
              return (
                <li
                  key={index + 1}
                  className="steper active"
                  onClick={() => {
                    setCurrentStep(index + 1);
                  }}
                >
                  {ele}
                </li>
              );
            } else {
              return (
                <li
                  key={index + 1}
                  className="steper"
                  onClick={() => {
                    setCurrentStep(index + 1);
                  }}
                >
                  {ele}
                </li>
              );
            }
          })}
          <li className="steper" onClick={() => next_step()}>
            Next
          </li>
        </ul>
      </div>
    </div>
  );
}

export default StepChooser;
