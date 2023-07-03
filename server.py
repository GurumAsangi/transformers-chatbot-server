from flask import Flask, jsonify, request
from transformers import pipeline

app = Flask(__name__)

@app.route('/question-answering', methods=['POST'])
def question_answering():
    data = request.get_json()
    context = "Albert Einstein was a German-born theoretical physicist. He developed the theory of relativity."
    question = data['question']
    
    # Load the question answering pipeline
    nlp = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    
    # Answer the question
    answer = nlp(question=question, context=context)
    
    # Return the answer as a JSON response
    return jsonify(answer)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
