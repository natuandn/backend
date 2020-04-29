import React from "react";
import { Link } from "react-router-dom";

function Home() {
  return (
    <div className="container">
      <div className="row">
        <div className="col s12 center offset">
          <h2 className="text-shadow">Want to create your own model ?</h2>
          <Link
            to="/main"
            className="btn-large waves-effect waves-light purple darken-3"
          >
            Get started now!
          </Link>
        </div>
        <div className="col s12 center offset">
          <h2 className="text-shadow">Test your model ?</h2>
          <Link
            to="/test"
            className="btn-large waves-effect waves-light green light-3"
          >
            Checkout here!
          </Link>
        </div>
      </div>
    </div>
  );
}

export default Home;
