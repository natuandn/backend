import React, { useState } from "react";

import StepOne from "./stepes/StepOne";
import StepTwo from "./stepes/StepTwo";
import StepThree from "./stepes/StepThree";
import StepChooser from "./stepes/StepChooser";
import StepTwoLoader from "./stepes/StepTwoLoader";

function Main() {
  const [currentStep, setCurrentStep] = useState(1);
  const [isReloadData, setIsReloadData] = useState(false);
  const [isLoadingStep2, setIsLoadingStep2] = useState(false);
  const next_step = () => {
    if (currentStep === 3) {
      return setCurrentStep(1);
    }
    setCurrentStep(currentStep + 1);
  };
  const previous_step = () => {
    if (currentStep === 1) {
      return setCurrentStep(3);
    }
    setCurrentStep(currentStep - 1);
  };
  return (
    <div className="container">
      <StepChooser
        previous_step={previous_step}
        currentStep={currentStep}
        next_step={next_step}
        setCurrentStep={setCurrentStep}
      />
      <div className="content-wraper">
        <div className={`step1 ${currentStep === 1 ? null : "hidden"}`}>
          <StepOne
            reloadData={value => {
              setIsLoadingStep2(true);
              setTimeout(() => {
                setIsReloadData(value);
                setIsLoadingStep2(false);
              }, 5500);
            }}
            isReloadData={isReloadData}
          />
        </div>

        <div className={`step2 ${currentStep === 2 ? null : "hidden"}`}>
          <StepTwo reloadData={isReloadData} isLoadingSteper={isLoadingStep2} />
        </div>

        <div className={`step3 ${currentStep === 3 ? null : "hidden"}`}>
          <StepThree />
        </div>
      </div>
    </div>
  );
}

export default Main;
