import React from "react";
import { render } from 'react-dom';
import ModelHistory from "./Components/ModelHistory";
import ModelAcc from "./Components/ModelAcc";
import CurrentModel from "./Components/CurrentModel";
import ModelStatus from "./Components/ModelStatus";
import '@progress/kendo-theme-default';
import axios from "axios";
import { Button } from "@progress/kendo-react-buttons";


function App() {
  const submitForm = (event) => {
    event.preventDefault();

    axios
      .post("http://127.0.0.1:8000/start/", {

      })
      .then((response) => {
        console.log(response)
        // successfully uploaded response
      })
      .catch((error) => {
        // error response
        console.log(error)
      });
  };
  return (
    <>
      <Button primary={true} onClick={submitForm}>Start Model</Button>
      {/* <div>
        <form onSubmit={submitForm}>
          <br />
          <input type="submit" />
        </form>
      </div> */}
      <div>
        <h3>
            Model Status
        </h3>
        {<ModelStatus />}
      </div>
      <div>
        <h3>
            Current Model Accuracy
        </h3>
        {<ModelAcc />}
      </div>
      <div>
        <h3>
            Current Prediction Model
        </h3>
        {<CurrentModel />}
      </div>
      <div>
        <h3>
            Training Model History
        </h3>
        {<ModelHistory />}
      </div>
 
    </>
  )
}

const rootElement = document.getElementById("root")
render(<App />, rootElement)
