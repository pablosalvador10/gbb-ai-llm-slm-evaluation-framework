import pandas as pd
import json
import tempfile
from azure.core.exceptions import ServiceResponseError
from typing import Any, Optional, Dict, Tuple
import os
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from azure.ai.evaluation import evaluate
from azure.ai.evaluation import QAEvaluator, ContentSafetyEvaluator
from azure.ai.projects import AIProjectClient
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential

# Set up logging
from my_utils.ml_logging import get_logger

logger = get_logger()

class AzureAIQualityEvaluator:
    def __init__(self,
                 azure_endpoint: Optional[str] = None,
                 api_key: Optional[str] = None,
                 azure_deployment: Optional[str] = None,
                 api_version: Optional[str] = None,
                 subscription_id: Optional[str] = None,
                 resource_group_name: Optional[str] = None,
                 project_name: Optional[str] = None,
                 azure_ai_foundry_connector: Optional[str] = None
                 ):
        """
        Initialize the AzureAIQualityEvaluator with model configuration and evaluators.

        :param azure_endpoint: Azure OpenAI endpoint.
        :param api_key: API key for Azure OpenAI.
        :param azure_deployment: Azure OpenAI deployment ID.
        :param api_version: API version for Azure OpenAI.
        :param subscription_id: Azure subscription ID.
        :param resource_group_name: Azure resource group name.
        :param project_name: Azure project name.
        """
        try:
            self.model_config = {
                "azure_endpoint": os.getenv("AZURE_AOAI_ENDPOINT") or azure_endpoint,
                "api_key": os.getenv("AZURE_AOAI_API_KEY") or api_key,
                "azure_deployment": os.getenv("AZURE_AOAI_COMPLETION_MODEL_DEPLOYMENT_ID") or azure_deployment,
                "api_version": os.getenv("AZURE_AOAI_DEPLOYMENT_VERSION") or api_version,
            }

            self.qa_evaluator = QAEvaluator(model_config=self.model_config, parallel=True)

            self.azure_ai_project = None
            if subscription_id and resource_group_name and project_name:
                self.azure_ai_project = {
                    "subscription_id": subscription_id,
                    "resource_group_name": resource_group_name,
                    "project_name": project_name
                }

            self.project_connection_string = os.getenv("AZURE_AI_FOUNDRY_CONNECTION_STRING")
            self.credential = DefaultAzureCredential()
            self.project = AIProjectClient.from_connection_string(
                conn_str=self.project_connection_string,
                credential=self.credential
            )
            
            self.content_safety_evaluator = ContentSafetyEvaluator(
               azure_ai_project=self.azure_ai_project, credential=self.credential, parallel=True
            )

            logger.info("AzureAIQualityEvaluator initialized successfully.")

        except Exception as e:
            logger.error(f"Failed to initialize AzureAIQualityEvaluator: {e}")
            raise

    def _convert_to_jsonl(self, data: pd.DataFrame, temp_file: tempfile.NamedTemporaryFile) -> None:
        """
        Convert DataFrame to JSONL format and write to a temporary file.

        :param data: DataFrame containing the data.
        :param temp_file: Temporary file to write the JSONL data.
        """
        try:
            for record in data.to_dict(orient='records'):
                json_record = json.dumps(record)
                temp_file.write(json_record + '\n')
            temp_file.flush()
            logger.info("Data successfully converted to JSONL format.")
        except Exception as e:
            logger.error(f"Error converting DataFrame to JSONL: {e}")
            raise

    def run_chat_quality(self, data_input: Any, 
                         azure_ai_project: Optional[Dict]=None) -> Any:
        """
        Evaluate the quality of chat responses using the QA evaluator.

        :param data_input: A pandas DataFrame or a path to a CSV file containing the data.
        :return: The result of the evaluation.
        """
        temp_file = None
        try:
            if isinstance(data_input, pd.DataFrame):
                data = data_input
            elif isinstance(data_input, str):
                data = pd.read_csv(data_input)
            else:
                raise ValueError("data_input must be a pandas DataFrame or a path to a CSV file.")

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl', mode='w')
            self._convert_to_jsonl(data, temp_file)

            result = evaluate(
                data=temp_file.name,
                evaluators={
                    "qa_evaluator": self.qa_evaluator,
                },
                evaluator_config={
                    "qa_evaluator": {
                        "question": "${data.question}",
                        "answer": "${data.answer}",
                        "context": "${data.context}",
                        "ground_truth": "${data.ground_truth}",
                    },
                },
                azure_ai_project=azure_ai_project or self.azure_ai_project
            )
            logger.info("Quality evaluation completed successfully.")
            metrics, studio_url = self.extract_chat_quality_results(result)
            if studio_url is not None:
                logger.info(f"See your results in the studio for more detailed information: {studio_url}")
            return metrics, studio_url
        except ServiceResponseError as e:
            logger.error(f"ServiceResponseError: {e}")
        except ConnectionResetError as e:
            logger.error(f"ConnectionResetError: {e}. The connection was forcibly closed by the remote host.")
        except Exception as e:
            logger.error(f"Error during content safety evaluation: {e}")
        finally:
            try:
                if temp_file:
                    os.remove(temp_file.name)
                    logger.info(f"Temporary file {temp_file.name} removed.")
            except Exception as e:
                logger.error(f"Error removing temporary file: {e}")

    def run_chat_content_safety(self, data_input: Any, 
                                azure_ai_project: Optional[Dict]=None) -> Any:
        """
        Evaluate the content safety of chat responses using the Content Safety evaluator.

        :param data_input: A pandas DataFrame or a path to a CSV file containing the data.
        :return: The result of the evaluation.
        """
        temp_file = None
        try:
            if isinstance(data_input, pd.DataFrame):
                data = data_input
            elif isinstance(data_input, str):
                data = pd.read_csv(data_input)
            else:
                raise ValueError("data_input must be a pandas DataFrame or a path to a CSV file.")

            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.jsonl', mode='w')
            self._convert_to_jsonl(data, temp_file)

            result = evaluate(
                data=temp_file.name,
                evaluators={
                    "content_safety_evaluator": self.content_safety_evaluator,
                },
                evaluator_config={
                    "content_safety_evaluator": {
                        "question": "${data.question}",
                        "answer": "${data.answer}",
                    },
                },
                azure_ai_project=azure_ai_project or self.azure_ai_project
            )
            logger.info("Content safety evaluation completed successfully.")
            metrics, studio_url = self.extract_chat_content_safety_results(result)
            if studio_url is not None:
                logger.info(f"See your results in the studio for more detailed information: {studio_url}")
            return metrics, studio_url
        except ServiceResponseError as e:
            logger.error(f"ServiceResponseError: {e}")
        except ConnectionResetError as e:
            logger.error(f"ConnectionResetError: {e}. The connection was forcibly closed by the remote host.")
        except Exception as e:
            logger.error(f"Error during content safety evaluation: {e}")
        finally:
            try:
                if temp_file:
                    os.remove(temp_file.name)
                    logger.info(f"Temporary file {temp_file.name} removed.")
            except Exception as e:
                logger.error(f"Error removing temporary file: {e}")

    @staticmethod
    def extract_chat_quality_results(result: Dict) -> Tuple[Dict, Optional[str]]:
        """
        Extract metrics and studio URL from chat quality evaluation results.

        :param result: The evaluation result dictionary.
        :return: A tuple containing the extracted metrics and the studio URL.
        """
        try:
            metrics = {key.replace('qa_evaluator.', ''): value for key, value in result.get('metrics', {}).items()}
            studio_url = result.get('studio_url', None)
            return metrics, studio_url
        except Exception as e:
            logger.error(f"Error extracting chat quality results: {e}")
            return {}, None
        

    @staticmethod
    def extract_chat_content_safety_results(result: Dict) -> Tuple[Dict, Optional[str]]:
        """
        Extract metrics and studio URL from chat content safety evaluation results.

        :param result: The evaluation result dictionary.
        :return: A tuple containing the extracted metrics and the studio URL.
        """
        try:
            metrics = {key.replace('content_safety_evaluator.', ''): value for key, value in result.get('metrics', {}).items()}
            studio_url = result.get('studio_url', None)
            return metrics, studio_url
        except Exception as e:
            logger.error(f"Error extracting chat content safety results: {e}")
            return {}, None


    def plot_metrics(self, metrics: Dict[str, float]):
        """
        Create an combined bar and line plot for the given metrics, with state-of-the-art visualization.

        :param metrics: Dictionary containing metric names and their values.
        """
        # Automatically create title based on deployment
        title = f"Metrics Overview for {self.model_config.azure_deployment}"

        # Convert metrics to DataFrame for easier manipulation
        metrics_df = pd.DataFrame(list(metrics.items()), columns=["Metric", "Score"])

        # Create a subplot that allows for a bar and line plot in the same figure
        fig = make_subplots(specs=[[{"secondary_y": True}]])

        # Add bar plot
        fig.add_trace(
            go.Bar(x=metrics_df["Metric"], y=metrics_df["Score"], name="Score (Bar)"),
            secondary_y=False,
        )

        # Add line plot
        fig.add_trace(
            go.Scatter(x=metrics_df["Metric"], y=metrics_df["Score"], name="Score (Line)", mode="lines+markers"),
            secondary_y=True,
        )

        # Add titles and labels
        fig.update_layout(
            title=title,
            xaxis_title="Metric",
            yaxis_title="Score",
            legend_title="Metric Type",
            template="plotly_white",
        )
        # Ensure y-axis starts at 0 and extends slightly above the maximum score to provide visual buffer
        max_score = metrics_df["Score"].max()
        y_axis_max = max_score + (max_score * 0.1) if max_score > 0 else 1  # Ensure there's always a range above 0

        fig.update_yaxes(title_text="Bar Score", secondary_y=False, range=[0, y_axis_max])
        fig.update_yaxes(title_text="Line Score", secondary_y=True, range=[0, y_axis_max])

        # Show the figure
        fig.show()