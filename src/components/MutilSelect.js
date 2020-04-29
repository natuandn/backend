import React from "react";
import Select from "react-select";
import makeAnimated from "react-select/animated";

const animatedComponents = makeAnimated();

function MutilSelect({ datas, placeholder, isMulti, handleSelectData }) {
  return (
    <div>
      <Select
        placeholder={placeholder}
        options={datas}
        components={animatedComponents}
        isMulti={isMulti}
        onChange={data => {
          handleSelectData(data);
        }}
      />
    </div>
  );
}

export default MutilSelect;
