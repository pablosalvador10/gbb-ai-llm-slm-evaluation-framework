# Building a Better Evaluation Framework for LLMs and SLMs

Hey there! You've stumbled upon our project where we're diving deep into the nuts and bolts of evaluating Generative AI applications, focusing on both Large and Smaller Language Models. This repo is our shared notebook, a place where we document our experiments, findings, and the technical challenges we tackle along the way. Using PromptFlow as our foundation, we're piecing together a framework that's all about getting hands-on and making sense of how to best evaluate and benchmark these complex AI systems. Join us in this technical exploration.

## 🤖 Challenges in LLM/SLM Evaluation

Evaluating LLMs and SLMs presents unique challenges, including the need for continuous evaluation, adherence to responsible AI practices, and the tailoring of evaluation metrics to specific applications. Prompt Flow addresses these challenges by offering:

- **Continuous Integration, Evaluation, and Deployment (CI/CE/CD)**: Implementing LLMOps for effective lifecycle management.
- **Responsible AI Practices**: Ensuring ethical use and mitigating potential risks.
- **Tailored Evaluation Metrics**: Customizing metrics for meaningful assessments.

## 💡 Why PromptFlow is Our Go-To for AI Evaluation

Our choice to integrate PromptFlow into our workflow was driven by its ability to cater to our specific evaluation needs. Here's a closer look at why it's our toolkit of choice:

- **Tailored Workflows**: PromptFlow's flexibility shines in its ability to let us craft evaluation workflows that are just right for our models. Whether it's offline analysis or real-time testing, we've got the tools we need to put our AI through its paces.

- **Comprehensive Testing**: The framework supports both offline and online evaluation strategies. This dual approach allows us to thoroughly vet our models in both controlled settings and live environments, ensuring they're up to any challenge.

- **Deep Dive Insights**: With PromptFlow's advanced tracing and observability, we're never in the dark about how our models are performing. Tracking every input and output gives us a granular view of our AI's behavior, making it easier to tweak, tune, and improve.

#### 🛠️ Implementing Your Evaluation Workflows

PromptFlow enables developers to define and manage evaluation workflows, automate prompt testing, and analyze outputs effectively. Follow our guides to implement your evaluation strategies:

1. **Define Evaluation Workflows**: Utilize Prompt Flow to set up comprehensive evaluation workflows.
2. **Automate Prompt Testing**: Leverage the framework to automate the testing of prompts and analyze outputs.
3. **Analyze and Optimize**: Use the insights gained from evaluations to debug, optimize, and improve your GenAI applications.

#### 🔍 Enhanced Tracing and Observability

With PromptFlow, developers gain enhanced tracing and observability features, allowing for detailed monitoring of GenAI applications from input to output. This includes:

- **Flexibility in Tracing**: Support for various endpoints, including Azure AI Foundry and Azure Application Insights.
- **Streamlined Deployment**: Deploy optimized GenAI applications to Azure AI Foundry for secure and scalable development.
- **Flex Flow**: Incorporate your applications into Prompt Flow for comprehensive evaluation and debugging.

#### 📊 Centralized Test History and Enhanced Analysis

PromptFlow's integration with Azure AI Foundry offers centralized test history, enhanced test analysis, and asset reutilization, facilitating:

- **Centralized Test History**: Store and track all historical tests for easy accessibility.
- **Enhanced Analysis**: Extract and visualize test results for comprehensive comparisons.
- **Asset Reutilization**: Streamline workflows by reusing previous test assets for efficiency.

## 🚀 How to Get Started

Before you begin, ensure you have the following:

- Access to Azure AI Foundry 

Let's get your development environment set up:

#### Configure Environment Variables 

Before running this notebook, you must configure certain environment variables to securely store our configuration. This practice helps in preventing sensitive data from being accidentally committed to version control systems.

Create a [`.env`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fpablosal%2FDesktop%2Fgbb-ai-llm-slm-evaluation-framework%2F.env%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "c:\Users\pablosal\Desktop\gbb-ai-llm-slm-evaluation-framework\.env") file in your project root (use the provided [`.env.sample`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fpablosal%2FDesktop%2Fgbb-ai-llm-slm-evaluation-framework%2F.env.sample%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "c:\Users\pablosal\Desktop\gbb-ai-llm-slm-evaluation-framework\.env.sample") as a template) and add the following variables:

```env
# Azure Open AI Completion Configuration
AZURE_AOAI_API_KEY=""
AZURE_AOAI_COMPLETION_MODEL_DEPLOYMENT_ID=""
AZURE_AOAI_ENDPOINT=""
AZURE_AOAI_DEPLOYMENT_VERSION=""
AZURE_AI_STUDIO_SUBSCRIPTION_ID=""
AZURE_AI_STUDIO_RESOURCE_GROUP_NAME=""
AZURE_AI_STUDIO_PROJECT_NAME=""
```

Please replace the placeholders with your actual Azure OpenAI and Azure AI Foundry configuration details:

- `AZURE_AOAI_API_KEY`: Your Azure OpenAI API key. You can obtain this from the Azure OpenAI service.
- `AZURE_AOAI_COMPLETION_MODEL_DEPLOYMENT_ID`: The deployment ID for your Azure OpenAI model.
- `AZURE_AOAI_ENDPOINT`: The endpoint URL for your Azure OpenAI service.
- `AZURE_AOAI_DEPLOYMENT_VERSION`: The version of your Azure OpenAI deployment.
- `AZURE_AI_STUDIO_SUBSCRIPTION_ID`: Your Azure subscription ID where the AI Studio project is hosted.
- `AZURE_AI_STUDIO_RESOURCE_GROUP_NAME`: The name of the resource group for your AI Studio project.
- `AZURE_AI_STUDIO_PROJECT_NAME`: The name of your AI Studio project.

To gather your Azure OpenAI API keys, visit the [Azure OpenAI service documentation](https://docs.microsoft.com/en-us/azure/cognitive-services/openai/). For the keys related to your project in Azure AI Foundry, you can find them in your project's settings within the Azure portal.

> 📌 **Note**
> Remember not to commit the .env file to your version control system. Add it to your .gitignore file to prevent it from being tracked.

#### Setting Up Conda Environment and Configuring VSCode for Jupyter Notebooks (Optional)

Follow these steps to create a Conda environment and set up your VSCode for running Jupyter Notebooks:

##### Create Conda Environment from the Repository

> Instructions for Windows users: 

1. **Create the Conda Environment**:
   - In your terminal or command line, navigate to the repository directory.
   - Execute the following command to create the Conda environment using the [`environment.yaml`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fpablosal%2FDesktop%2Fgbb-ai-llm-slm-evaluation-framework%2Fenvironment.yaml%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "c:\Users\pablosal\Desktop\gbb-ai-llm-slm-evaluation-framework\environment.yaml") file:
     ```bash
     conda env create -f environment.yaml
     ```
   - This command creates a Conda environment as defined in [`environment.yaml`](command:_github.copilot.openRelativePath?%5B%7B%22scheme%22%3A%22file%22%2C%22authority%22%3A%22%22%2C%22path%22%3A%22%2Fc%3A%2FUsers%2Fpablosal%2FDesktop%2Fgbb-ai-llm-slm-evaluation-framework%2Fenvironment.yaml%22%2C%22query%22%3A%22%22%2C%22fragment%22%3A%22%22%7D%5D "c:\Users\pablosal\Desktop\gbb-ai-llm-slm-evaluation-framework\environment.yaml").

2. **Activating the Environment**:
   - After creation, activate the new Conda environment by using:
     ```bash
     conda activate promptflow-eval-framework
     ```

> Instructions for Linux users (or Windows users with WSL or other linux setup): 

1. **Use `make` to Create the Conda Environment**:
   - In your terminal or command line, navigate to the repository directory and look at the Makefile.
   - Execute the `make` command specified below to create the Conda environment using the `environment.yaml` file:
     ```bash
     make create_conda_env
     ```

2. **Activating the Environment**:
   - After creation, activate the new Conda environment by using:
     ```bash
     conda activate promptflow-eval-framework
     ```

##### Configure VSCode for Jupyter Notebooks

1. **Install Required Extensions**:
   - Download and install the `Python` and `Jupyter` extensions for VSCode. These extensions provide support for running and editing Jupyter Notebooks within VSCode.

2. **Open the Notebook**:
   - Open the Jupyter Notebook file (`01-promptflow-evaluation-howto.ipynb`) in VSCode.

3. **Attach Kernel to VSCode**:
   - After creating the Conda environment, it should be available in the kernel selection dropdown. This dropdown is located in the top-right corner of the VSCode interface.
   - Select your newly created environment (`promptflow-eval-framework`) from the dropdown. This sets it as the kernel for running your Jupyter Notebooks.

4. **Run the Notebook**:
   - Once the kernel is attached, you can run the notebook by clicking on the "Run All" button in the top menu, or by running each cell individually.


By following these steps, you'll establish a dedicated Conda environment for your project and configure VSCode to run Jupyter Notebooks efficiently. This environment will include all the necessary dependencies specified in your `environment.yaml` file. If you wish to add more packages or change versions, please use `pip install` in a notebook cell or in the terminal after activating the environment, and then restart the kernel. The changes should be automatically applied after the session restarts.

## 📚 Resources

- **Prompt Flow Documentation**: For detailed information on Prompt Flow and its components, visit our [Documentation]().
- **Tutorials**: Check out our [Tutorials]() for hands-on guides on setting up and utilizing Prompt Flow for LLM/SLM evaluation.

### Disclaimer
> [!IMPORTANT]
> This software is provided for demonstration purposes only. It is not intended to be relied upon for any purpose. The creators of this software make no representations or warranties of any kind, express or implied, about the completeness, accuracy, reliability, suitability or availability with respect to the software or the information, products, services, or related graphics contained in the software for any purpose. Any reliance you place on such information is therefore strictly at your own risk.