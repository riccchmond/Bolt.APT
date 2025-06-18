import requests
import json
import time
from typing import Dict, Optional
from datetime import datetime, timezone


class FeedbackProcessor:
    def __init__(self,
                 api_url: str = "http://localhost:1234/v1/chat/completions",
                 max_retries: int = 3,
                 timeout: int = 480):
        """
        Initialize the FeedbackProcessor with LM Studio's local API endpoint.

        Args:
            api_url (str): URL of the LM Studio API endpoint
            max_retries (int): Maximum connection attempts
            timeout (int): Timeout in seconds for API requests
        """
        self.api_url = api_url
        self.max_retries = max_retries
        self.timeout = timeout
        self._print_init_status()

    def _print_init_status(self):
        """Log initialization details."""
        print(f"Initializing FeedbackProcessor at {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
        print(f"Using API endpoint: {self.api_url}")
        print(f"Timeout set to: {self.timeout} seconds")

    def check_connection(self) -> bool:
        """
        Verify connection to LM Studio API.

        Returns:
            bool: True if connection successful
        """
        try:
            print("Checking connection to LM Studio...")
            response = requests.get(
                self.api_url.replace("/v1/chat/completions", ""),
                timeout=5
            )
            return response.status_code == 200
        except requests.exceptions.ConnectionError:
            print("Connection failed - LM Studio may not be running")
            return False
        except requests.exceptions.Timeout:
            print("Connection check timed out")
            return False

    def generate_coaching_feedback(self, analysis_report: Dict) -> Optional[str]:
        """
        Generate human-readable coaching feedback from biomechanical analysis.

        Args:
            analysis_report (dict): Phase-based biomechanical analysis data

        Returns:
            str: Formatted coaching feedback or None if failed
        """
        if not self.check_connection():
            self._print_connection_troubleshooting()
            return None

        prompt = self._create_prompt(analysis_report)
        return self._process_api_request(prompt)

    def _create_prompt(self, analysis_report: Dict) -> str:
        """
        Create structured prompt for phase-based analysis feedback.
        """
        prompt_lines = [
            "Analyze the following technical flaws and provide simple, actionable feedback for each phase:",
            ""
        ]

        for phase, report in analysis_report.items():
            prompt_lines.append(f"Phase: {phase}")
            significant_deviations = False

            for metric, data in report.items():
                if self._is_significant_deviation(data):
                    significant_deviations = True
                    prompt_lines.append(self._format_deviation(metric, data))

            if not significant_deviations:
                prompt_lines.append("- No significant deviations detected.")
            prompt_lines.append("")

        return "\n".join(prompt_lines)

    def _is_significant_deviation(self, data: Dict) -> bool:
        """Check if deviation exceeds threshold."""
        deviation = data.get('deviation')
        return deviation is not None and abs(deviation) > 5

    def _format_deviation(self, metric: str, data: Dict) -> str:
        """Format individual deviation entry."""
        deviation = data['deviation']
        calc_value = f"{data['calculated']:.1f}°" if data['calculated'] is not None else "N/A"
        ideal_value = f"{data['ideal']:.1f}°" if data['ideal'] is not None else "N/A"
        direction = "lower" if deviation < 0 else "higher"
        return (f"- {metric}: {abs(deviation):.1f}° {direction} than ideal "
                f"({calc_value} vs recommended {ideal_value})")

    def _process_api_request(self, prompt: str) -> Optional[str]:
        """Handle API communication with retries and error handling."""
        for attempt in range(self.max_retries):
            try:
                print(f"API Attempt {attempt + 1}/{self.max_retries}...")
                start_time = time.time()

                response = requests.post(
                    self.api_url,
                    headers={"Content-Type": "application/json"},
                    json={
                        "messages": [
                            {
                                "role": "system",
                                "content": "You are an expert athletic coach specializing in biomechanical analysis."
                            },
                            {"role": "user", "content": prompt}
                        ],
                        "temperature": 0.7,
                        "max_tokens": 300,
                        "stream": False
                    },
                    timeout=self.timeout
                )

                print(f"Request completed in {time.time() - start_time:.2f}s")

                if response.status_code == 200:
                    return response.json()['choices'][0]['message']['content']

                print(f"API Error {response.status_code}: {response.text}")

            except requests.exceptions.Timeout:
                print(f"Timeout after {self.timeout}s")
                if attempt < self.max_retries - 1:
                    time.sleep(2)
            except requests.exceptions.RequestException as e:
                print(f"Request failed: {str(e)}")
                if attempt < self.max_retries - 1:
                    time.sleep(2)

        self._print_failure_instructions()
        return None

    def _print_connection_troubleshooting(self):
        """Display connection troubleshooting guide."""
        print("\nConnection troubleshooting:")
        print("1. Ensure LM Studio is running")
        print("2. Verify Local Server settings:")
        print("   - Server enabled")
        print("   - Port 1234 active")
        print("3. Check model is loaded")
        print("4. Restart LM Studio if needed")

    def _print_failure_instructions(self):
        """Display final failure instructions."""
        print("\nAll attempts failed. Check:")
        print("- LM Studio availability")
        print("- System resource usage")
        print("- Model compatibility")