import argparse
import pandas as pd
import requests
import json
import random
import time
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from rouge_score import rouge_scorer
import nltk
import matplotlib.pyplot as plt
from vector import fetch_data_from_table
from nltk.corpus import stopwords
from collections import Counter


try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('stopwords')


API_URL = "http://127.0.0.1:8181"

class RAGEvaluator:
    def __init__(self, sample_size=10, api_url=API_URL):
        self.sample_size = sample_size
        self.api_url = api_url
        self.scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
        self.smoothing = SmoothingFunction().method1
        try:
            self.stopwords = set(stopwords.words('french'))
        except:
            self.stopwords = set() 
        
    def get_random_samples(self):
        """Retrieves a random sample of documents from Cloud SQL"""
        print("Retrieving data from Cloud SQL...")
        documents = fetch_data_from_table()
        
        if not documents or len(documents) == 0:
            raise ValueError("No documents found in the database")

        if len(documents) < self.sample_size:
            print(f"Warning: only {len(documents)} documents available, less than the requested {self.sample_size}")
            self.sample_size = len(documents)

        samples = random.sample(documents, self.sample_size)
        return samples
    
    def generate_questions(self, documents):
        """Generates questions based on document content with focused reference answers"""
        print("Generating questions from documents...")
        questions = []
        
        for doc in documents:
            content = doc.page_content
            response = requests.post(
                f"{self.api_url}/query",
                json={
                    "text": f"Based on this text, generate: 1) a single factual question whose answer is in the text, and 2) a concise reference answer to that question (1-2 sentences). Format as JSON with 'question' and 'answer' fields: {content[:500]}",
                    "top_k": 3,
                    "model": "gemini-pro", 
                    "language": "Français"
                }
            )
            
            if response.status_code == 200:
                result = response.json()["answer"].strip()
                try:
                    qa_pair = json.loads(result)
                    generated_question = qa_pair.get("question", "")
                    reference_answer = qa_pair.get("answer", "")
                    
                    if not reference_answer or len(reference_answer) < 10:

                        extract_response = requests.post(
                            f"{self.api_url}/query",
                            json={
                                "text": f"Extract a short passage (1-3 sentences) from this text that answers the question: '{generated_question}'. Only return the extracted passage: {content}",
                                "top_k": 3,
                                "model": "gemini-pro", 
                                "language": "Français"
                            }
                        )
                        if extract_response.status_code == 200:
                            reference_answer = extract_response.json()["answer"].strip()
                        else:
                            reference_answer = content  
                    
                    questions.append({
                        "document": doc,
                        "question": generated_question,
                        "expected_answer": reference_answer,
                        "full_content": content  
                    })
                except:

                    generated_question = result
                    
                    extract_response = requests.post(
                        f"{self.api_url}/query",
                        json={
                            "text": f"Extract a short passage (1-3 sentences) from this text that answers the question: '{generated_question}'. Only return the extracted passage: {content}",
                            "top_k": 3,
                            "model": "gemini-pro", 
                            "language": "Français"
                        }
                    )
                    if extract_response.status_code == 200:
                        reference_answer = extract_response.json()["answer"].strip()
                    else:
                        reference_answer = content
                        
                    questions.append({
                        "document": doc,
                        "question": generated_question,
                        "expected_answer": reference_answer,
                        "full_content": content
                    })
            else:
                print(f"Error generating question: {response.text}")
        
        return questions
    
    def query_rag_system(self, question, language="Français"):
        """Queries the RAG system via the API"""
        try:
            response = requests.post(
                f"{self.api_url}/query",
                json={
                    "text": question,
                    "top_k": 3,
                    "model": "gemini-pro",
                    "language": language
                }
            )
            
            if response.status_code != 200:
                print(f"API Error: {response.status_code} - {response.text}")
                return None, None
                
            result = response.json()
            return result["answer"], result["sources"]
        except Exception as e:
            print(f"Error querying the API: {str(e)}")
            return None, None
    
    def evaluate_retrieval(self, question, sources, expected_doc):
        """Evaluates retrieval quality with more lenient metrics"""
        expected_content = expected_doc.page_content
        retrieved_contents = [source["content"] for source in sources]
        

        contains_expected = False
        overlap_scores = []
        keyword_match_scores = []

        question_keywords = self.extract_keywords(question)
        
        for content in retrieved_contents:

            overlap = self.text_overlap(expected_content, content)
            overlap_scores.append(overlap)
            

            keywords_score = self.keyword_match_score(content, question_keywords)
            keyword_match_scores.append(keywords_score)

            if overlap > 0.3 or keywords_score > 0.5:  
                contains_expected = True
        
        metrics = {
            "max_overlap": max(overlap_scores) if overlap_scores else 0,
            "avg_overlap": sum(overlap_scores) / len(overlap_scores) if overlap_scores else 0,
            "max_keyword_match": max(keyword_match_scores) if keyword_match_scores else 0,
            "contains_expected": contains_expected
        }
        
        return metrics
    
    def extract_keywords(self, text):
        """Extracts important keywords from text"""

        tokens = nltk.word_tokenize(text.lower())

        keywords = [token for token in tokens if token not in self.stopwords and len(token) > 3]

        freq_dist = Counter(keywords)

        return [word for word, count in freq_dist.most_common(5)]
    
    def keyword_match_score(self, text, keywords):
        """Calculates how well a text matches a set of keywords"""
        if not keywords:
            return 0
            
        tokens = set(nltk.word_tokenize(text.lower()))
        matches = sum(1 for keyword in keywords if keyword in tokens)
        
        return matches / len(keywords)
    
    def text_overlap(self, text1, text2):
        """Calculates overlap between two texts with a more lenient approach"""
        tokens1 = set(nltk.word_tokenize(text1.lower()))
        tokens2 = set(nltk.word_tokenize(text2.lower()))
        
        if not tokens1 or not tokens2:
            return 0

        important_tokens1 = [t for t in tokens1 if t not in self.stopwords and len(t) > 3]
        important_tokens2 = [t for t in tokens2 if t not in self.stopwords and len(t) > 3]

        overlap_standard = len(tokens1.intersection(tokens2)) / min(len(tokens1), len(tokens2))

        if important_tokens1 and important_tokens2:
            overlap_important = len(set(important_tokens1).intersection(set(important_tokens2))) / min(len(important_tokens1), len(important_tokens2))

            return max(overlap_standard, overlap_important)
        
        return overlap_standard
    
    def evaluate_generation(self, generated_answer, expected_answer, full_content=None):
        """Evaluates the quality of the generated answer with more lenient metrics"""
        if not generated_answer:
            return {
                "rouge1": 0,
                "rouge2": 0,
                "rougeL": 0,
                "bleu": 0,
                "keyword_recall": 0
            }

        sentences = []
        if expected_answer:
            sentences.extend(nltk.sent_tokenize(expected_answer))
        if full_content:
            sentences.extend(nltk.sent_tokenize(full_content))
        
        direct_rouge_scores = self.scorer.score(expected_answer, generated_answer)

        best_rouge1 = direct_rouge_scores["rouge1"].fmeasure
        best_rouge2 = direct_rouge_scores["rouge2"].fmeasure
        best_rougeL = direct_rouge_scores["rougeL"].fmeasure
        
        for sentence in sentences:
            if len(sentence) > 20: 
                sentence_scores = self.scorer.score(sentence, generated_answer)
                best_rouge1 = max(best_rouge1, sentence_scores["rouge1"].fmeasure)
                best_rouge2 = max(best_rouge2, sentence_scores["rouge2"].fmeasure)
                best_rougeL = max(best_rouge2, sentence_scores["rougeL"].fmeasure)
        
        #BLEU score with more weight on unigram matches
        reference = [nltk.word_tokenize(expected_answer.lower())]
        hypothesis = nltk.word_tokenize(generated_answer.lower())
        
        try:
            weights = (0.7, 0.15, 0.1, 0.05)
            bleu_score = sentence_bleu(reference, hypothesis, weights=weights, smoothing_function=self.smoothing)
        except Exception:
            bleu_score = 0
            
        expected_keywords = self.extract_keywords(expected_answer)
        keyword_recall = self.keyword_match_score(generated_answer, expected_keywords)
        
        return {
            "rouge1": best_rouge1,
            "rouge2": best_rouge2,
            "rougeL": best_rougeL,
            "bleu": bleu_score,
            "keyword_recall": keyword_recall
        }
        
    def run_evaluation(self):
        """Runs the complete evaluation"""
        try:
            health_check = requests.get(f"{self.api_url}/health")
            if health_check.status_code != 200:
                print(f"⚠️ API is not accessible at {self.api_url}. Check that the server is running.")
                return []

            samples = self.get_random_samples()

            test_cases = self.generate_questions(samples)
            
            print(f"Evaluating {len(test_cases)} questions...")
            
            results = []
            
            for i, test_case in enumerate(tqdm(test_cases)):
                question = test_case["question"]
                expected_doc = test_case["document"]
                expected_answer = test_case["expected_answer"]
                full_content = test_case.get("full_content", expected_doc.page_content)
                
                print(f"\nQuestion {i+1}: {question}")
                print(f"Expected answer: {expected_answer[:100]}..." if len(expected_answer) > 100 else f"Expected answer: {expected_answer}")
                
                start_time = time.time()
                
                answer, sources = self.query_rag_system(question)

                response_time = time.time() - start_time
                
                if not answer or not sources:
                    print(f"Query failed for question {i+1}")
                    continue
                
                print(f"Answer: {answer[:100]}..." if len(answer) > 100 else f"Answer: {answer}")

                retrieval_metrics = self.evaluate_retrieval(question, sources, expected_doc)

                generation_metrics = self.evaluate_generation(answer, expected_answer, full_content)
                
                results.append({
                    "question": question,
                    "answer": answer,
                    "expected_answer": expected_answer,
                    "expected_doc": expected_doc.page_content[:300] + "...",
                    "sources": sources,
                    "retrieval_metrics": retrieval_metrics,
                    "generation_metrics": generation_metrics,
                    "latency": response_time
                })

            self.calculate_and_report_metrics(results)
            
            return results
            
        except Exception as e:
            print(f"Error during evaluation: {str(e)}")
            import traceback
            traceback.print_exc()
            return []
    
    def calculate_and_report_metrics(self, results):
        """Calculates and displays overall metrics"""
        if not results:
            print("No results to analyze")
            return
            
        # Retrieval metrics
        avg_max_overlap = np.mean([r["retrieval_metrics"]["max_overlap"] for r in results])
        avg_avg_overlap = np.mean([r["retrieval_metrics"]["avg_overlap"] for r in results])
        avg_keyword_match = np.mean([r["retrieval_metrics"]["max_keyword_match"] for r in results])
        retrieval_success_rate = np.mean([1 if r["retrieval_metrics"]["contains_expected"] else 0 for r in results])
        
        # Generation metrics
        avg_rouge1 = np.mean([r["generation_metrics"]["rouge1"] for r in results])
        avg_rouge2 = np.mean([r["generation_metrics"]["rouge2"] for r in results])
        avg_rougeL = np.mean([r["generation_metrics"]["rougeL"] for r in results])
        avg_bleu = np.mean([r["generation_metrics"]["bleu"] for r in results])
        avg_keyword_recall = np.mean([r["generation_metrics"]["keyword_recall"] for r in results])
        
        # Average latency
        avg_latency = np.mean([r["latency"] for r in results])

        print("\n" + "="*50)
        print("EVALUATION RESULTS")
        print("="*50)
        
        print("\nRetrieval Metrics:")
        print(f"- Average maximum overlap: {avg_max_overlap:.4f}")
        print(f"- Average overlap: {avg_avg_overlap:.4f}")
        print(f"- Average keyword match: {avg_keyword_match:.4f}")
        print(f"- Retrieval success rate: {retrieval_success_rate:.2%}")
        
        print("\nGeneration Metrics:")
        print(f"- Average ROUGE-1 score: {avg_rouge1:.4f}")
        print(f"- Average ROUGE-2 score: {avg_rouge2:.4f}")
        print(f"- Average ROUGE-L score: {avg_rougeL:.4f}")
        print(f"- Average BLEU score: {avg_bleu:.4f}")
        print(f"- Average keyword recall: {avg_keyword_recall:.4f}")
        
        print(f"\nAverage latency: {avg_latency:.2f} seconds")

        self.save_results_to_csv(results)

        self.generate_visualizations(results)
    
    def save_results_to_csv(self, results):
        """Saves results to a CSV file"""
        df_rows = []
        
        for i, r in enumerate(results):
            row = {
                "question_id": i + 1,
                "question": r["question"],
                "answer": r["answer"],
                "expected_answer": r["expected_answer"],
                "max_overlap": r["retrieval_metrics"]["max_overlap"],
                "avg_overlap": r["retrieval_metrics"]["avg_overlap"],
                "keyword_match": r["retrieval_metrics"]["max_keyword_match"],
                "contains_expected": r["retrieval_metrics"]["contains_expected"],
                "rouge1": r["generation_metrics"]["rouge1"],
                "rouge2": r["generation_metrics"]["rouge2"],
                "rougeL": r["generation_metrics"]["rougeL"],
                "bleu": r["generation_metrics"]["bleu"],
                "keyword_recall": r["generation_metrics"]["keyword_recall"],
                "latency": r["latency"]
            }
            df_rows.append(row)
        
        df = pd.DataFrame(df_rows)
        
        output_file = f"evaluation_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        df.to_csv(output_file, index=False)
        print(f"\nDetailed results saved to {output_file}")
        
        with open(f"evaluation_qa_{time.strftime('%Y%m%d_%H%M%S')}.txt", "w", encoding="utf-8") as f:
            for i, r in enumerate(results):
                f.write(f"===== Question {i+1} =====\n")
                f.write(f"Q: {r['question']}\n\n")
                f.write(f"Expected A: {r['expected_answer']}\n\n")
                f.write(f"Generated A: {r['answer']}\n\n")
                f.write(f"Sources used: {len(r['sources'])}\n\n")
                f.write("-" * 80 + "\n\n")
    
    def generate_visualizations(self, results):
        """Generates visualizations of the results"""
        retrieval_metrics = [r["retrieval_metrics"]["max_overlap"] for r in results]
        keyword_match = [r["retrieval_metrics"]["max_keyword_match"] for r in results]
        rouge1_scores = [r["generation_metrics"]["rouge1"] for r in results]
        rouge2_scores = [r["generation_metrics"]["rouge2"] for r in results]
        rougeL_scores = [r["generation_metrics"]["rougeL"] for r in results]
        bleu_scores = [r["generation_metrics"]["bleu"] for r in results]
        keyword_recall = [r["generation_metrics"]["keyword_recall"] for r in results]
        latencies = [r["latency"] for r in results]
        question_ids = [i+1 for i in range(len(results))]
        
        fig, axes = plt.subplots(3, 2, figsize=(15, 15))
        
        #retrieval overlap chart
        axes[0, 0].bar(question_ids, retrieval_metrics, color='skyblue')
        axes[0, 0].set_title('Maximum Overlap by Question')
        axes[0, 0].set_xlabel('Question ID')
        axes[0, 0].set_ylabel('Overlap')
        axes[0, 0].grid(axis='y', linestyle='--', alpha=0.7)
        
        #keyword match chart
        axes[0, 1].bar(question_ids, keyword_match, color='lightgreen')
        axes[0, 1].set_title('Keyword Match Score by Question')
        axes[0, 1].set_xlabel('Question ID')
        axes[0, 1].set_ylabel('Score')
        axes[0, 1].grid(axis='y', linestyle='--', alpha=0.7)
        
        #rouge scores chart
        axes[1, 0].plot(question_ids, rouge1_scores, 'o-', label='ROUGE-1')
        axes[1, 0].plot(question_ids, rouge2_scores, 's-', label='ROUGE-2')
        axes[1, 0].plot(question_ids, rougeL_scores, '^-', label='ROUGE-L')
        axes[1, 0].set_title('ROUGE Scores by Question')
        axes[1, 0].set_xlabel('Question ID')
        axes[1, 0].set_ylabel('Score')
        axes[1, 0].legend()
        axes[1, 0].grid(linestyle='--', alpha=0.7)
        
        #bleu scores chart
        axes[1, 1].bar(question_ids, bleu_scores, color='lightgreen')
        axes[1, 1].set_title('BLEU Scores by Question')
        axes[1, 1].set_xlabel('Question ID')
        axes[1, 1].set_ylabel('Score')
        axes[1, 1].grid(axis='y', linestyle='--', alpha=0.7)
        
        #keyword recall chart
        axes[2, 0].bar(question_ids, keyword_recall, color='orange')
        axes[2, 0].set_title('Keyword Recall by Question')
        axes[2, 0].set_xlabel('Question ID')
        axes[2, 0].set_ylabel('Score')
        axes[2, 0].grid(axis='y', linestyle='--', alpha=0.7)
        
        #response time chart
        axes[2, 1].bar(question_ids, latencies, color='salmon')
        axes[2, 1].set_title('Response Time by Question')
        axes[2, 1].set_xlabel('Question ID')
        axes[2, 1].set_ylabel('Time (s)')
        axes[2, 1].grid(axis='y', linestyle='--', alpha=0.7)
        
        plt.tight_layout()

        output_file = f"evaluation_visualizations_{time.strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(output_file)
        print(f"Visualizations saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description='RAG System Evaluation')
    parser.add_argument('--samples', type=int, default=10, help='Number of samples to evaluate')
    parser.add_argument('--api-url', type=str, default=API_URL, help='RAG API URL')
    
    args = parser.parse_args()
    
    print(f"Starting evaluation with {args.samples} samples...")
    print(f"API URL: {args.api_url}")
    
    try:
        evaluator = RAGEvaluator(sample_size=args.samples, api_url=args.api_url)
        evaluator.run_evaluation()
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()