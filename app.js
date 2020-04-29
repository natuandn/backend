const express = require("express");
const app = express();
const cors = require("cors");
const ngrok = require("ngrok");
const dotenv = require("dotenv");

(async () => {
  dotenv.config();
  app.use(cors());
  app.use(express.json());
  app.use(express.urlencoded({ extended: false }));

  app.use("/", require("./routes/mainRoutes"));
  app.use("/", require("./routes/fileRoutes"));

  const port = process.env.PORT || 3000;
  app.listen(port, err => {
    if (err) throw err;
    console.log(`app is listen on port ${port}`);
  });
  //TUAN22
  
})();
