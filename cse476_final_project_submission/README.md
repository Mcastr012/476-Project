This project answers the released test questions and saves them in the JSON format needed for submission.

To run it offline without SOL, go into the project folder and run:

```bash
cd /Users/marcocastro/Desktop/476-Project/cse476_final_project_submission
python3 -m pip install --user python-dotenv
python3 generate_answer_template.py --backend nearest
```

This offline mode does not need SOL, VPN, or an API key. It runs through the full question file and writes a full answers JSON, but it is mainly for testing that the code works from start to finish.

If you want to run the real model version, make a `.env` file in this folder with:

```env
OPENAI_API_KEY=your_key_here
API_BASE=https://openai.rc.asu.edu/v1
MODEL_NAME=qwen3-30b-a3b-instruct-2507
```

Then run:

```bash
python3 generate_answer_template.py
```

This is the real model run.

If a run stops in the middle, you can continue with:

```bash
python3 generate_answer_template.py --resume
```
