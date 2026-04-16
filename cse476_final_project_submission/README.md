This project is a reasoning agent that answers the test questions and saves the results in the Json format the assignment needs. generate_answer_template.py runs everything, reasoning_agent.py has the main logic, the test questions are in cse_476_final_project_test_data.json and the output goes into cse_476_final_project_answers.json.

To run it, I need to be on the ASU network or VPN, have a SOL API key, and make a .env file with the key, API base, and model name. I also need the dev data file available in the right location.

The agent uses different methods depending on the question type, like code generation for coding questions.