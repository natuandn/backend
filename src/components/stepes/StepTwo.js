import React from "react";

import { FontAwesomeIcon } from "@fortawesome/react-fontawesome";
import { faArrowRight } from "@fortawesome/free-solid-svg-icons";
import { configModel, getListCsvFile } from "../../helpers/callApi";
import MutilSelect from "../MutilSelect";
import StepTwoLoader from "./StepTwoLoader";
function StepTwo({ reloadData, isLoadingSteper }) {
  const [fetchFilenames, setFetchFilenames] = React.useState([]);
  const [fetchHeaders, setFetchHeaders] = React.useState([]);
  const [fetchFeatureLength, setFetchFeatureLength] = React.useState([]);
  const [headerSets, setHeaderSetS] = React.useState([]);
  const [index, setIndex] = React.useState(null);

  const [filenameSelected, setFilenameSelected] = React.useState(null);
  const [featureSelected, setFeatureSelected] = React.useState(null);
  const [outputSelected, setOuputSelected] = React.useState(null);
  const [outputShape, setOutputShape] = React.useState(null);
  const [errMsg, setErrMsg] = React.useState("");
  const [successMsg, setSuccessMsg] = React.useState("");

  React.useEffect(() => {
    const getContent = async () => {
      try {
        const res = await getListCsvFile();
        const { data } = res;
        const fetchFilename = [];
        const fetchHeader = [];
        const fetchFeatureLength = [];
        const dataArray = JSON.parse(data);

        dataArray.map((file, i) => {
          fetchFilename.push({
            label: file.filename,
            value: file.filename
          });
          fetchHeader.push(file.header);
          fetchFeatureLength.push(file.uniq);
          return {};
        });
        setFetchHeaders(fetchHeader);
        setFetchFilenames(fetchFilename);
        setFetchFeatureLength(fetchFeatureLength);
      } catch (err) {
        console.log(err);
      }
    };
    getContent();
  }, [reloadData]);

  const handleSubmit = async e => {
    e.preventDefault();
    setErrMsg("");
    setSuccessMsg("");
    if (outputShape !== null && featureSelected !== null) {
      const featureHeader = featureSelected.map(feature => feature.value),
        outputHeader = outputSelected.value,
        outputCount = outputShape,
        filename = filenameSelected.value;

      let data = { filename, featureHeader, outputHeader, outputCount };
      try {
        const res = await configModel(data);
        setSuccessMsg(res.data.msg);
      } catch (err) {
        if (err.response) {
          if (err.response.data.configErr) {
            setErrMsg(err.response.data.configErr);
          }
          if (err.response.data.details) {
            setErrMsg(err.response.data.details[0].message);
          }
          if (err.response.data.fileError) {
            setErrMsg(err.response.data.fileError);
          }
        }
      }
    } else {
      setErrMsg(`You must fill all field first!!!`);
    }
  };
  /**
   * @param {filename selected} value
   * todo : Update new feature list aka filename and set index to get config data in next field
   */
  const handleFileNameSelected = value => {
    let index = fetchFilenames.indexOf(value);
    let header = fetchHeaders[index];
    const features = header.map(header => {
      return {
        label: header,
        value: header
      };
    });
    setFilenameSelected(value);
    setHeaderSetS(features);
    setIndex(index);
  };

  const handleOutputSelected = value => {
    // set outputheader value
    setOuputSelected(value);

    // set feature header shape
    let uniqFeature = fetchFeatureLength[index];
    let valueIndex = headerSets.indexOf(value);

    setOutputShape(Number.parseInt(uniqFeature[valueIndex]));
  };

  return (
    <div className="row">
      {isLoadingSteper === false ? (
        <form className="col s12" action="" onSubmit={handleSubmit}>
          <div className="row">
            <div className="input-field col l6 s12 m6">
              <MutilSelect
                placeholder={"Select file"}
                datas={fetchFilenames}
                isMulti={false}
                handleSelectData={handleFileNameSelected}
              />
            </div>
            <div className="input-field col l6 s12 m6">
              <MutilSelect
                placeholder={"Select feature header"}
                datas={headerSets}
                isMulti={true}
                handleSelectData={header => {
                  setFeatureSelected(header);
                }}
              />
            </div>
          </div>

          <div className="row">
            <div className="input-field col l6 s12 m6">
              <MutilSelect
                placeholder={"Select output header"}
                datas={headerSets.filter(ele => {
                  if (featureSelected) {
                    return !featureSelected.includes(ele);
                  } else {
                    return ele;
                  }
                })}
                isMulti={false}
                handleSelectData={handleOutputSelected}
              />
            </div>
            <div className="col l6 s12 m6 center pt-30">
              <button
                className="btn purple darken-3 waves-effect waves-light "
                type="submit"
                style={{ zIndex: "0" }}
              >
                Submit
                <FontAwesomeIcon icon={faArrowRight} />
              </button>
            </div>
          </div>
          <div className="row">
            <div className="col l12 s12 m12">
              {successMsg ? (
                <div className="col s12 center">
                  <b className="green-text ">{successMsg}</b>
                </div>
              ) : null}{" "}
              {errMsg ? (
                <div className="col s12 center">
                  <b className="red-text ">{errMsg}</b>
                </div>
              ) : null}
            </div>
          </div>
        </form>
      ) : (
        <StepTwoLoader />
      )}
    </div>
  );
}

export default StepTwo;
