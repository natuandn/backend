import React from "react";
import { CopyToClipboard } from "react-copy-to-clipboard";
import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faCopy } from "@fortawesome/free-solid-svg-icons";
import { csvUpload } from "../../helpers/callApi";
function StepOne({ isReloadData, reloadData }) {
  const [selectedFile, setSelectedFile] = React.useState(null);
  const [fileName, setFileName] = React.useState("Select your file!");
  const [uploadMsg, setUploadMsg] = React.useState("");
  const [errMsg, setErrMsg] = React.useState("");
  const [fileToken, setFileToken] = React.useState("");
  const [copied, setCopied] = React.useState(false);

  const handleSubmit = async e => {
    e.preventDefault();
    setErrMsg("");
    if (selectedFile != null) {
      let data = new FormData(document.getElementById("uploadForm"));
      
      try {
        const res = await csvUpload(data);
        console.log(data);
        
        const { msg, file_name } = res.data;
        setUploadMsg(msg);
        setFileToken(file_name);
        reloadData(!isReloadData);
      } catch (err) {
        console.log(err.response);
        setErrMsg(err.response.data);
      }
    } else {
      setErrMsg("Select your file first !");
    }
  };

  const handleFile = e => {
    setFileName(e.target.files[0].name);
    setSelectedFile({
      selectedFile: e.target.files[0],
      loaded: 0
    });
  };

  return (
    <div className="row">
      <form
        className="col s12 form-upload"
        onSubmit={handleSubmit}
        encType="multipart/form-data"
        id="uploadForm"
      >
        <div className="file-upload-wrapper" data-text={fileName}>
          <input
            type="file"
            name="myfile"
            id="myfile"
            accept=".csv, application/vnd.ms-excel"
            onChange={handleFile}
          />
          <button className="btn-upload waves-effect waves-light">
            Upload
          </button>
        </div>
      </form>

      {uploadMsg ? (
        <div className="pt-20">
          <h6 className="center green-text pt-20">{uploadMsg}</h6>
        </div>
      ) : null}
      {errMsg ? (
        <div className="pt-20">
          <h6 className="center red-text pt-20">{errMsg}</h6>
        </div>
      ) : null}
      {fileToken ? (
        <div className="copy-wrapper">
          <div className="file-token">{fileToken}</div>
          <CopyToClipboard text={fileToken} onCopy={() => setCopied(true)}>
            <button className="btn-copy">
              <FontAwesomeIcon icon={faCopy} />
            </button>
          </CopyToClipboard>
          {copied ? <span style={{ color: "red" }}>Copied.</span> : null}
        </div>
      ) : null}
    </div>
  );
}

export default StepOne;
