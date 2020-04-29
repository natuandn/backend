import React from "react";
import { modelUpload, predictModel } from "../helpers/callApi";

function ModelTest() {
  const [fileName, setFileName] = React.useState(
    "Select your JSON model file!"
  );
  const [errMsg, setErrMsg] = React.useState("");
  const [resMsg, setResMsg] = React.useState("");
  const [predictResult, setPredictResult] = React.useState([]);
  const handleSubmitFile = async e => {
    e.preventDefault();
    setErrMsg("");
    setResMsg("");
    if (fileName !== "Select your JSON model file!") {
      const data = new FormData(document.getElementById(`modelForm`));
      try {
        const res = await modelUpload(data);

        setResMsg(res.data.msg);
      } catch (err) {
        console.log(err);
        setErrMsg(err.response.data.msg);
      }
    }
  };
  const handlePredictModel = async e => {
    e.preventDefault();
    const input = e.target.input.value;
    try {
      if (input === "") {
        setErrMsg("Input should not empty");
      } else {
        const res = await predictModel(input);
        setPredictResult(res.data);
      }
    } catch (err) {
      console.log(err);
    }
  };

  const handleFileChange = e => {
    if (e.target.files.length > 1) {
      setFileName(e.target.files[0].name + "-" + e.target.files[1].name);
    } else {
      setFileName(e.target.files[0].name);
    }
  };
  return (
    <div className="container">
      <div className="card z-depth-3 row">
        <div className="card-content col s12 m12 l12">
          <span className="card-title center">Test your model</span>
          <form
            onSubmit={handleSubmitFile}
            className="row"
            encType="multipart/form-data"
            id="modelForm"
          >
            <div
              className="file-upload-wrapper col s12 m4 l6 offset-l3 mb-20"
              data-text={fileName}
            >
              <input
                type="file"
                name="myFile"
                id="myFile"
                onChange={handleFileChange}
                multiple
              />
            </div>

            <div className="col s12 center">
              <button type="submit" className="btn waves-effect waves-light">
                Upload
              </button>
            </div>
          </form>
          <div className="row">
            {errMsg ? (
              <div className="col s12 center red-text">{errMsg}</div>
            ) : null}
            {resMsg ? (
              <div className="col s12 center green-text">{resMsg}</div>
            ) : null}
          </div>
          <div className="card-action center">
            <form onSubmit={handlePredictModel} className="row">
              <div className="input-field col s12 ">
                <input type="text" name="input" id="input"></input>
                <label htmlFor="input">Type your input with space</label>
              </div>
              <button
                className="btn waves-effect waves-light center purple darken-2"
                name="action"
              >
                Predict
              </button>
            </form>
            <div className="row">
              <div className="col s12 center purple-text">
                {predictResult.length ? JSON.stringify(predictResult) : null}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default ModelTest;
