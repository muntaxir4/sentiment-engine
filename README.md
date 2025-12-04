# Sentiment Extraction


The project uses deepseek-r1:1.5 as the base llm model to extract sentiments out from text and next.js 16, a react based framework for the web ui to interact with the model.

## Steps to setup the project
#### Setting up the backend (model)
1. Install ollama in your system from https://ollama.com/download
2. Pull the `deepseek-r1:1.5b` model using the comand `ollama pull deepseek-r1:1.5b`.
3. To make sure the model has been pulled,  run `ollama run deepseek-r1:1.5b` . If everything is right, you should be able to interact with the model locally.
4. While being inside the `backend` folder, run `ollama create -f Modelfile sentiment-engine`. This command creates a model with defined configuration in the `Modelfile` on top of `deepseek-r1:1.5b` with the name of `sentiment-engine`.
5. If everything is right, you should be able to interact with this new model by running `ollama run sentiment-engine`

Great we're done with setting up the **backend**!


#### Setting up the frontend.
1. cd into the `frontend` directory `cd frontend`.
2. run `npm i --legacy-peer-deps` to install all the packages. (Make sure you have node and npm installed in your system).
3. run `npm run dev`. This should start a local server at port `3000`.
4. If everything is right you should be able to interact with the model on http://localhost:3000



## Contributing
1. fork the repository.
2. create a new branch and implement your changes.
3. merge your changes to `main`.
4. create a pull request.
