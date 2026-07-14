import json
import os
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim
import streamlit as st

@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


class SemanticAnalyzer:

    def __init__(self):
        self.model = load_model()
        
        # Hardcoded fallback dictionary to guarantee app never crashes if JSON file fails
        self.reference = {
            "Data Structures": "A data structure is a storage that is used to store and organize data. It is a way of arranging data on a computer so that it can be accessed and updated efficiently. Examples include arrays, linked lists, stacks, queues, trees, and graphs.",
            "Object Oriented Programming": "Object-oriented programming (OOP) is a computer programming model that organizes software design around data, or objects, rather than functions and logic. It rests on pillars like inheritance, polymorphism, encapsulation, and abstraction.",
            "Database Management Systems": "A Database Management System (DBMS) is software used to store, retrieve, and run queries on data. A DBMS serves as an interface between an end-user and a database, allowing users to create, read, update, and delete data in the database.",
            "Operating Systems": "An Operating System (OS) is system software that manages computer hardware, software resources, and provides common services for computer programs. Examples include Microsoft Windows, macOS, Linux, and Android.",
            "Computer Networks": "A computer network is a set of computers sharing resources located on or provided by network nodes. The computers use common communication protocols over digital interconnections to communicate with each other."
        }

        # Dynamically load from external JSON file if available to allow easy overrides
        try:
            json_path = os.path.join(
                os.path.dirname(__file__),
                "..",
                "data",
                "reference_concepts.json"
            )
            if os.path.exists(json_path):
                with open(json_path, 'r', encoding='utf-8') as file:
                    file_data = json.load(file)
                    if isinstance(file_data, dict):
                        self.reference.update(file_data)
        except Exception:
            pass  # Suppress errors and gracefully rely on the built-in fallback dictionary

    def get_reference(self, topic):
        return self.reference.get(
            topic,
            "Reference concept not available."
        )

    def analyze(self, topic, transcript):
        reference_text = self.reference.get(topic)

        if reference_text is None:
            raise ValueError(
                f"No reference found for '{topic}'"
            )

        # --------------------------------------
        # Generate Embeddings
        # --------------------------------------
        reference_embedding = self.model.encode(
            reference_text,
            convert_to_tensor=True
        )

        transcript_embedding = self.model.encode(
            transcript,
            convert_to_tensor=True
        )

        similarity = cos_sim(
            reference_embedding,
            transcript_embedding
        )

        similarity_score = round(
            float(similarity[0][0]) * 100,
            2
        )

        # Ensure bounds are within standard percentage scale limits
        similarity_score = max(0.0, min(100.0, similarity_score))

        # --------------------------------------
        # Determine Evaluation Metrics & Feedback
        # --------------------------------------
        if similarity_score >= 85:
            feedback = "Excellent Understanding"
            confidence = "High"
            strengths = [
                "Accurately explained the core principles of the concept.",
                "Used clear technical terms seamlessly."
            ]
            improvements = [
                "Keep up the great explanation styling structure."
            ]
            recommendation = (
                "The student has completely mastered this module topic area block successfully."
            )

        elif similarity_score >= 70:
            feedback = "Good Understanding"
            confidence = "Medium"
            strengths = [
                "Covered the foundational definitions properly."
            ]
            improvements = [
                "Incorporate a few more live deployment use-case examples.",
                "Elaborate deeper on internal mechanics."
            ]
            recommendation = (
                "Slight revision of advanced sub-concepts recommended for better concept coverage."
            )

        elif similarity_score >= 60:
            feedback = "Basic Understanding"
            confidence = "Low"
            strengths = [
                "Shows initial understanding."
            ]
            improvements = [
                "Review the topic.",
                "Explain concepts step-by-step.",
                "Include definitions and examples."
            ]
            recommendation = (
                "Revise the topic and practice speaking again."
            )

        else:
            feedback = "Needs Improvement"
            confidence = "Very Low"
            strengths = [
                "Attempted the explanation."
            ]
            improvements = [
                "Study the topic thoroughly.",
                "Practice before recording.",
                "Cover the main definition first.",
                "Use simple examples."
            ]
            recommendation = (
                "Significant improvement is required. Review the topic and try again."
            )

        return {
            "score": similarity_score,
            "feedback": feedback,
            "confidence": confidence,
            "strengths": strengths,
            "improvements": improvements,
            "recommendation": recommendation
        }