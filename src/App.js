import React from "react";
import { Route, BrowserRouter, Switch } from "react-router-dom";
import Home from "./components/Home";
import Main from "./components/Main";
import ModelTest from "./components/ModelTest";

function App() {
  return (
    
    <div className="App">
      <React.Fragment>
        <h1 style={{color: "red", textAlign: 'center', textTransform: 'uppercase'}}>R-IoT Experiment</h1>
      </React.Fragment>
      <BrowserRouter>
      
        <Switch>
          <Route exact path="/" component={Home} />
          <Route exact path="/main" component={Main} />
          <Route exact path="/test" component={ModelTest} />
        </Switch>
      </BrowserRouter>
    </div>
  );
}

export default App;
