# Set compatibility flags before ANY imports
import os
import sys

# Force Pydantic v1 compatibility
os.environ["PYDANTIC_V1"] = "1"
os.environ["USE_PYDANTIC_V1"] = "true" 
os.environ["PYDANTIC_V1_COMPAT"] = "true"

import json
import firebase_admin
from firebase_admin import credentials, firestore

# Import with better error handling
try:
    from langchain_google_vertexai import ChatVertexAI
    from langchain_core.prompts import ChatPromptTemplate
    LANGCHAIN_AVAILABLE = True
    print("LangChain imports successful")
except Exception as e:
    print(f"LangChain import failed: {e}")
    ChatVertexAI = None
    ChatPromptTemplate = None
    LANGCHAIN_AVAILABLE = False

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any, List, Optional

app = FastAPI()

# Add CORS middleware with flexible origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserIdRequest(BaseModel):
    user_id: str

class ProcessingResponse(BaseModel):
    success: bool
    message: str
    user_id: str
    generated_goal: Optional[str] = None
    roadmap: Optional[Dict[str, Any]] = None
    error_details: Optional[str] = None

class AssessmentResponse(BaseModel):
    success: bool
    message: str
    user_id: str
    assessments_generated: Optional[int] = None
    assessments_kept: Optional[int] = None
    assessments_removed: Optional[int] = None
    total_assessments: Optional[int] = None
    error_details: Optional[str] = None

# Initialize Firebase once when the module loads
def initialize_firebase():
    """Initialize Firebase Admin SDK (only once)"""
    try:
        # Check if Firebase is already initialized
        try:
            firebase_admin.get_app()
            print("Firebase already initialized")
            return True
        except ValueError:
            # Firebase not initialized, so initialize it
            pass
        
        # Method 1: Environment variable with JSON content (RENDER)
        firebase_key = os.environ.get("FIREBASE_SERVICE_ACCOUNT_KEY")
        if firebase_key:
            try:
                service_account_info = json.loads(firebase_key)
                cred = credentials.Certificate(service_account_info)
                firebase_admin.initialize_app(cred)
                print("Firebase initialized with environment variable")
                return True
            except Exception as e:
                print(f"Failed to initialize with env var: {e}")
        
        # Method 2: Default credentials (works on GCP)
        try:
            firebase_admin.initialize_app()
            print("Firebase initialized with default credentials")
            return True
        except Exception as e:
            print(f"Failed to initialize with default credentials: {e}")
        
        # Method 3: Local development fallback
        local_paths = [
            "pacific-vault-470814-s5-25cf6513ecef.json",
            "modell/pacific-vault-470814-s5-25cf6513ecef.json",
            r"modell\pacific-vault-470814-s5-25cf6513ecef.json"
        ]
        
        for local_path in local_paths:
            if os.path.exists(local_path):
                cred = credentials.Certificate(local_path)
                firebase_admin.initialize_app(cred)
                print(f"Firebase initialized with local credentials: {local_path}")
                return True
        
        print("All Firebase initialization methods failed")
        return False
        
    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        return False

# Initialize Firebase when the module loads
firebase_initialized = initialize_firebase()
if firebase_initialized:
    try:
        db = firestore.client()
        print("Firestore client initialized")
    except Exception as e:
        print(f"Failed to initialize Firestore client: {e}")
        db = None
        firebase_initialized = False
else:
    db = None

def get_user_skills(user_id: str) -> Dict[str, Any]:
    """Retrieves the skills map from a user document."""
    if not db:
        raise Exception("Firebase not initialized")
    
    if not user_id or not user_id.strip():
        raise Exception("user_id cannot be empty")
    
    try:
        user_doc_ref = db.collection('user').document(user_id.strip())
        user_doc_snapshot = user_doc_ref.get()

        if user_doc_snapshot.exists:
            user_data = user_doc_snapshot.to_dict()
            print("Retrieved user data:", user_data)
            
            # Check for both "skills" and "skills " to handle data inconsistency
            if 'skills' in user_data:
                skills_data = user_data['skills']
            elif 'skills ' in user_data:
                skills_data = user_data['skills ']
            else:
                skills_data = {}

            # Ensure skills_data has the expected structure
            if not isinstance(skills_data, dict):
                skills_data = {}
            
            # Ensure primary and secondary are lists
            if 'primary' not in skills_data:
                skills_data['primary'] = []
            if 'secondary' not in skills_data:
                skills_data['secondary'] = []
                
            if not isinstance(skills_data['primary'], list):
                skills_data['primary'] = []
            if not isinstance(skills_data['secondary'], list):
                skills_data['secondary'] = []

            return skills_data
        else:
            raise Exception(f"No user document found with ID '{user_id}'")

    except Exception as e:
        raise Exception(f"Error retrieving user skills: {str(e)}")

def goal_generator(skills_data: dict) -> str:
    """Generates a career goal based on user's comprehensive skills."""
    
    if not skills_data:
        raise Exception("Skills data is empty or invalid")
    
    primary_skills = skills_data.get('primary', [])
    secondary_skills = skills_data.get('secondary', [])
    all_skills = primary_skills + secondary_skills
    
    if not LANGCHAIN_AVAILABLE:
        # Enhanced fallback logic based on skill combinations
        skills_str = ' '.join(all_skills).lower()
        
        # More sophisticated skill analysis for fallback
        if any(skill in skills_str for skill in ['python', 'django', 'flask']) and any(skill in skills_str for skill in ['javascript', 'react', 'vue', 'angular']):
            return "Full Stack Developer specializing in Python and JavaScript"
        elif any(skill in skills_str for skill in ['data', 'analytics', 'sql', 'python', 'pandas', 'numpy']):
            return "Data Analyst with Python expertise"
        elif any(skill in skills_str for skill in ['machine learning', 'ai', 'tensorflow', 'pytorch']):
            return "Machine Learning Engineer"
        elif any(skill in skills_str for skill in ['devops', 'docker', 'kubernetes', 'aws', 'cloud']):
            return "DevOps Engineer specializing in Cloud Technologies"
        elif any(skill in skills_str for skill in ['ui', 'ux', 'design', 'figma', 'adobe']):
            return "UI/UX Designer with technical skills"
        elif 'python' in skills_str:
            return "Python Developer"
        elif any(skill in skills_str for skill in ['javascript', 'react', 'vue', 'angular', 'frontend']):
            return "Frontend Developer"
        else:
            # Create goal based on skill combination
            return f"Technology Specialist in {', '.join(primary_skills[:2])}" if primary_skills else "Technology Professional"

    try:
        # Use environment variable for project ID if available
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "pacific-vault-470814-s5")
        
        llm = ChatVertexAI(
            model="gemini-2.0-flash-exp", 
            temperature=0.3, 
            max_output_tokens=256, 
            project=project_id
        )

        # Create comprehensive skills description
        formatted_skills = f"Primary Skills: {', '.join(primary_skills)}. Secondary Skills: {', '.join(secondary_skills)}."

        prompt = ChatPromptTemplate([
            ("system", 
            "You are an expert career advisor. Analyze ALL the user's skills comprehensively to suggest a specific and realistic career goal. "
            "Consider how their primary and secondary skills complement each other and can be leveraged together. "
            "The goal should be achievable within 1-2 years and should integrate multiple skills where possible. "
            "Focus on roles that utilize skill combinations rather than just single skills. "
            "Provide only the career goal as a concise statement (e.g., 'Full Stack Developer with Data Analytics focus', 'DevOps Engineer with Python automation expertise', etc.). "
            "Do not include explanations, just the career goal title."),
            ("user", "Based on my complete skill set: {skills}, what integrated career goal should I pursue that leverages multiple skills?"),
        ])

        chain = prompt | llm
        response = chain.invoke({"skills": formatted_skills})
        career_goal = response.content.strip().strip('"\'')
        print(f"Generated Career Goal: {career_goal}")
        return career_goal
    except Exception as e:
        print(f"LangChain failed, using fallback: {e}")
        # Enhanced fallback logic
        skills_str = ' '.join(all_skills).lower()
        
        if any(skill in skills_str for skill in ['python', 'django', 'flask']) and any(skill in skills_str for skill in ['javascript', 'react', 'vue']):
            return "Full Stack Developer specializing in Python and JavaScript"
        elif any(skill in skills_str for skill in ['data', 'analytics', 'sql']) and 'python' in skills_str:
            return "Data Analyst with Python expertise"
        else:
            return f"Technology Specialist in {', '.join(primary_skills[:2])}" if primary_skills else "Technology Professional"

def roadmap_generator(skills_data: dict, user_goal: str) -> dict:
    """Generates a comprehensive career roadmap based on ALL user skills and their goal."""
    
    if not skills_data or not user_goal:
        raise Exception("Skills data or user goal is empty or invalid")

    primary_skills = skills_data.get('primary', [])
    secondary_skills = skills_data.get('secondary', [])
    all_skills = primary_skills + secondary_skills

    if not LANGCHAIN_AVAILABLE:
        # Enhanced fallback roadmap based on skill combinations
        skills_str = ' '.join(all_skills).lower()
        
        # Create more comprehensive fallback roadmap
        tasks = []
        
        # Foundation phase
        tasks.append({
            "status": "current",
            "taskName": f"Strengthen foundational skills in {', '.join(primary_skills[:2])}",
            "timeAllocation": "4-6 weeks",
            "resources": [{"title": "Online Learning Platform", "link": "https://www.coursera.org"}]
        })
        
        # Integration phase
        if len(primary_skills) > 1:
            tasks.append({
                "status": "upcoming", 
                "taskName": f"Integrate {primary_skills[0]} with {primary_skills[1] if len(primary_skills) > 1 else secondary_skills[0] if secondary_skills else 'complementary skills'}",
                "timeAllocation": "6-8 weeks",
                "resources": [{"title": "Project-based Learning", "link": "https://github.com"}]
            })
        
        # Practical application
        tasks.append({
            "status": "upcoming",
            "taskName": f"Build portfolio projects showcasing multiple skills",
            "timeAllocation": "8-10 weeks", 
            "resources": [{"title": "Portfolio Examples", "link": "https://github.com"}]
        })
        
        # Advanced skills
        if secondary_skills:
            tasks.append({
                "status": "upcoming",
                "taskName": f"Develop expertise in {', '.join(secondary_skills[:2])}",
                "timeAllocation": "6-8 weeks",
                "resources": [{"title": "Advanced Courses", "link": "https://www.udemy.com"}]
            })
        
        return {
            "title": f"Comprehensive Roadmap to become {user_goal}",
            "tasks": tasks
        }

    try:
        # Use environment variable for project ID if available
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "pacific-vault-470814-s5")
        
        llm = ChatVertexAI(
            model="gemini-2.0-flash-exp", 
            temperature=0.2, 
            max_output_tokens=1024, 
            project=project_id
        )

        # Create comprehensive skills description
        formatted_skills = f"Primary Skills: {', '.join(primary_skills)}. Secondary Skills: {', '.join(secondary_skills)}. All Skills Combined: {', '.join(all_skills)}."

        prompt = ChatPromptTemplate([
            ("system", 
            "You are an expert career and skill roadmap generator. "
            "Generate a comprehensive roadmap with 4-6 key tasks that leverages ALL the user's skills (both primary and secondary). "
            "The roadmap should show how to integrate multiple skills and build upon their complete skill set to reach their goal. "
            "Consider skill synergies and how different skills complement each other. "
            "Return ONLY a valid JSON object with NO additional text, explanations, or markdown formatting. "
            "Use this exact structure: "
            '{{"title": "string describing the comprehensive roadmap", '
            '"tasks": [{{"status": "completed/current/upcoming", "taskName": "specific task that integrates multiple skills", "timeAllocation": "time estimate", "resources": [{{"title": "resource name", "link": "https://valid-url.com"}}]}}]}}. '
            'Each task should reference or build upon multiple skills where possible. '
            'Progress from foundational skill integration to advanced skill combinations. '
            'Ensure all JSON values are properly quoted strings and the JSON is complete and valid.'),
            ("user", "Based on ALL my skills: {skills}, generate a comprehensive roadmap to become: {goal}. "
                     "Make sure to leverage the synergy between my primary and secondary skills."),
        ])

        chain = prompt | llm
        response = chain.invoke({"skills": formatted_skills, "goal": user_goal})
        
        # More robust JSON extraction
        raw_content = response.content.strip()
        print(f"Raw LLM Response: {raw_content}")
        
        # Remove any markdown code blocks or extra formatting
        raw_content = raw_content.replace('json', '').replace('', '').strip()
        
        # Find the start and end of the JSON object
        json_start = raw_content.find('{')
        json_end = raw_content.rfind('}')
        
        if json_start != -1 and json_end != -1:
            clean_json_str = raw_content[json_start:json_end + 1]
        else:
            raise Exception("No valid JSON found in LLM response")
        
        print(f"Extracted JSON: {clean_json_str}")
        
        # Parse the JSON
        roadmap = json.loads(clean_json_str)
        
        # Validate the roadmap structure
        if not isinstance(roadmap, dict):
            raise Exception("Roadmap is not a dictionary")
        
        if "title" not in roadmap or "tasks" not in roadmap:
            raise Exception("Roadmap missing required fields (title, tasks)")
        
        if not isinstance(roadmap.get("tasks"), list):
            raise Exception("Tasks must be a list")
        
        if len(roadmap.get("tasks", [])) == 0:
            raise Exception("Roadmap must contain at least one task")
        
        print("Roadmap validation successful")
        return roadmap
        
    except json.JSONDecodeError as e:
        print(f"JSON parsing error, using enhanced fallback: {e}")
        # Return enhanced fallback roadmap
        return {
            "title": f"Comprehensive Roadmap to become {user_goal}",
            "tasks": [
                {
                    "status": "current",
                    "taskName": f"Master core competencies in {', '.join(primary_skills[:2])}", 
                    "timeAllocation": "4-6 weeks",
                    "resources": [{"title": "Skill Integration Course", "link": "https://www.coursera.org"}]
                },
                {
                    "status": "upcoming",
                    "taskName": f"Build projects combining {primary_skills[0] if primary_skills else 'primary'} with {secondary_skills[0] if secondary_skills else 'secondary'} skills",
                    "timeAllocation": "6-8 weeks",
                    "resources": [{"title": "Project Portfolio Guide", "link": "https://github.com"}]
                },
                {
                    "status": "upcoming",
                    "taskName": "Develop advanced expertise across your skill stack",
                    "timeAllocation": "8-10 weeks",
                    "resources": [{"title": "Advanced Learning Path", "link": "https://www.udemy.com"}]
                }
            ]
        }
    except Exception as e:
        print(f"LangChain roadmap generation failed, using enhanced fallback: {e}")
        # Return enhanced fallback roadmap with skill integration
        skills_summary = f"{', '.join(primary_skills[:2])} and {', '.join(secondary_skills[:2])}" if primary_skills and secondary_skills else ', '.join(all_skills[:3])
        
        return {
            "title": f"Multi-Skill Roadmap to become {user_goal}",
            "tasks": [
                {
                    "status": "current",
                    "taskName": f"Strengthen foundation in {skills_summary}",
                    "timeAllocation": "4-6 weeks",
                    "resources": [{"title": "Comprehensive Learning", "link": "https://www.coursera.org"}]
                },
                {
                    "status": "upcoming",
                    "taskName": "Create integrated projects using multiple skills",
                    "timeAllocation": "6-8 weeks",
                    "resources": [{"title": "Multi-Skill Projects", "link": "https://github.com"}]
                },
                {
                    "status": "upcoming",
                    "taskName": "Build professional portfolio showcasing skill combinations",
                    "timeAllocation": "4-6 weeks",
                    "resources": [{"title": "Portfolio Building", "link": "https://github.com"}]
                }
            ]
        }

def assessment_generator(skill: str, category: str) -> dict:
    """Generates an assessment in JSON format for a given skill + category"""
    
    if not skill or not category:
        raise Exception("Skill and category cannot be empty")
    
    if not LANGCHAIN_AVAILABLE:
        # Return mock assessment
        return create_mock_assessment(skill, category)

    try:
        # Use environment variable for project ID if available
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "pacific-vault-470814-s5")
        
        llm = ChatVertexAI(
            model="gemini-2.0-flash-exp",
            temperature=0.3,
            max_output_tokens=2048,
            project=project_id
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are an assessment generator that creates technical skill assessments. "
             "Generate exactly 10 multiple-choice questions for the given skill. "
             "Return a valid JSON object with this exact structure:\n"
             '{{\n'
             '  "skill": string,\n'
             '  "category": string,\n'
             '  "questions": [\n'
             '    {{\n'
             '      "questionId": string (format: "q1", "q2", etc.),\n'
             '      "questionText": string,\n'
             '      "options": [string, string, string, string] (exactly 4 options),\n'
             '      "correctOption": string (must match one of the options exactly)\n'
             '    }}\n'
             '  ]\n'
             '}}\n\n'
             "Important rules:\n"
             "- Generate exactly 10 questions\n"
             "- Each question must have exactly 4 options (A, B, C, D format)\n"
             "- correctOption must be the full text of the correct answer, not just A/B/C/D\n"
             "- Questions should test practical knowledge and real-world application\n"
             "- Make questions progressively challenging (easy to advanced)\n"
             "- Ensure JSON is valid and properly formatted"),
            ("user", "Generate a 10-question assessment for skill: {skill}, category: {category}. "
                     "Focus on practical, industry-relevant questions that test real understanding.")
        ])

        chain = prompt | llm
        response = chain.invoke({"skill": skill, "category": category})

        raw_content = response.content.strip()
        print(f"Raw assessment response for {skill}: {raw_content[:200]}...")
        
        # Extract JSON safely
        json_start = raw_content.find('{')
        json_end = raw_content.rfind('}')
        clean_json_str = raw_content[json_start:json_end+1] if json_start != -1 else raw_content

        assessment = json.loads(clean_json_str)
        
        # Validate the assessment structure
        if validate_assessment_structure(assessment, skill, category):
            return assessment
        else:
            print(f"Assessment validation failed for {skill}, using fallback")
            return create_mock_assessment(skill, category)
            
    except json.JSONDecodeError as e:
        print(f"JSON parsing error for {skill}, using fallback: {e}")
        return create_mock_assessment(skill, category)
    except Exception as e:
        print(f"Failed to generate assessment for {skill}, using fallback: {e}")
        return create_mock_assessment(skill, category)

def create_mock_assessment(skill: str, category: str) -> dict:
    """Create a mock assessment when LangChain is unavailable"""
    return {
        "skill": skill,
        "category": category,
        "questions": [
            {
                "questionId": "q1",
                "questionText": f"What is a fundamental concept in {skill}?",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correctOption": "Option A"
            },
            {
                "questionId": "q2", 
                "questionText": f"Which best practice applies to {skill}?",
                "options": ["Practice 1", "Practice 2", "Practice 3", "Practice 4"],
                "correctOption": "Practice 1"
            },
            {
                "questionId": "q3",
                "questionText": f"How do you implement {skill} in real projects?",
                "options": ["Method 1", "Method 2", "Method 3", "Method 4"],
                "correctOption": "Method 1"
            },
            {
                "questionId": "q4",
                "questionText": f"What tool is commonly used with {skill}?",
                "options": ["Tool A", "Tool B", "Tool C", "Tool D"],
                "correctOption": "Tool A"
            },
            {
                "questionId": "q5",
                "questionText": f"Which is an advanced technique in {skill}?",
                "options": ["Technique 1", "Technique 2", "Technique 3", "Technique 4"],
                "correctOption": "Technique 1"
            },
            {
                "questionId": "q6",
                "questionText": f"What is a common mistake when learning {skill}?",
                "options": ["Mistake A", "Mistake B", "Mistake C", "Mistake D"],
                "correctOption": "Mistake A"
            },
            {
                "questionId": "q7",
                "questionText": f"How do you debug issues in {skill}?",
                "options": ["Debug A", "Debug B", "Debug C", "Debug D"],
                "correctOption": "Debug A"
            },
            {
                "questionId": "q8",
                "questionText": f"What framework works well with {skill}?",
                "options": ["Framework 1", "Framework 2", "Framework 3", "Framework 4"],
                "correctOption": "Framework 1"
            },
            {
                "questionId": "q9",
                "questionText": f"Which performance optimization applies to {skill}?",
                "options": ["Optimization A", "Optimization B", "Optimization C", "Optimization D"],
                "correctOption": "Optimization A"
            },
            {
                "questionId": "q10",
                "questionText": f"What is the future trend for {skill}?",
                "options": ["Trend 1", "Trend 2", "Trend 3", "Trend 4"],
                "correctOption": "Trend 1"
            }
        ]
    }

def validate_assessment_structure(assessment, expected_skill, expected_category):
    """Validate that the generated assessment matches the expected structure"""
    try:
        # Check basic structure
        if not isinstance(assessment, dict):
            return False
        
        # Check required fields
        if "questions" not in assessment:
            return False
        
        questions = assessment["questions"]
        if not isinstance(questions, list) or len(questions) != 10:
            return False
        
        # Validate each question
        for question in questions:
            if not isinstance(question, dict):
                return False
            
            required_fields = ["questionId", "questionText", "options", "correctOption"]
            for field in required_fields:
                if field not in question:
                    return False
            
            # Validate options
            options = question["options"]
            if not isinstance(options, list) or len(options) != 4:
                return False
            
            # Validate correctOption is one of the options
            if question["correctOption"] not in options:
                return False
        
        return True
        
    except Exception:
        return False

def get_existing_assessments(user_id: str) -> List[Dict[str, Any]]:
    """Get existing assessments for a user"""
    if not db:
        raise Exception("Firebase not initialized")
    
    try:
        user_ref = db.collection('user').document(user_id)
        user_doc = user_ref.get()
        
        if user_doc.exists:
            user_data = user_doc.to_dict()
            assessments = user_data.get('assessment', [])
            
            # Ensure assessments is a list
            if not isinstance(assessments, list):
                return []
            
            return assessments
        else:
            return []
            
    except Exception as e:
        raise Exception(f"Error retrieving existing assessments: {str(e)}")

def create_skill_key(skill: str, category: str) -> str:
    """Create a unique key for skill + category combination"""
    return f"{skill.strip().lower()}_{category.strip().lower()}"

def initialize_user_assessment_array(user_id: str):
    """Initialize the assessment field as an empty array if it doesn't exist"""
    if not db:
        raise Exception("Firebase not initialized")
    
    try:
        user_ref = db.collection('user').document(user_id)
        user_doc = user_ref.get()
        
        if user_doc.exists:
            user_data = user_doc.to_dict()
            if 'assessment' not in user_data or not isinstance(user_data.get('assessment'), list):
                user_ref.update({"assessment": []})
                print(f"Initialized assessment array for user {user_id}")
        else:
            raise Exception(f"User document {user_id} does not exist")
            
    except Exception as e:
        raise Exception(f"Error initializing assessment array: {str(e)}")

def sync_assessments_with_skills(user_id: str) -> Dict[str, Any]:
    """Sync assessments with current skills"""
    print(f"\nSyncing assessments for user: {user_id}")
    
    # Initialize assessment array
    initialize_user_assessment_array(user_id)
    
    # Get current user skills
    skills_data = get_user_skills(user_id)
    primary_skills = skills_data.get('primary', [])
    secondary_skills = skills_data.get('secondary', [])
    
    print(f"Primary skills ({len(primary_skills)}): {primary_skills}")
    print(f"Secondary skills ({len(secondary_skills)}): {secondary_skills}")
    
    # Create set of current skills with their categories
    current_skill_keys = set()
    current_skills_map = {}
    
    for skill in primary_skills:
        if skill and skill.strip():  # Skip empty skills
            key = create_skill_key(skill, "primary")
            current_skill_keys.add(key)
            current_skills_map[key] = {"skill": skill.strip(), "category": "primary"}
    
    for skill in secondary_skills:
        if skill and skill.strip():  # Skip empty skills
            key = create_skill_key(skill, "secondary")
            current_skill_keys.add(key)
            current_skills_map[key] = {"skill": skill.strip(), "category": "secondary"}
    
    # Get existing assessments
    existing_assessments = get_existing_assessments(user_id)
    print(f"Found {len(existing_assessments)} existing assessments")
    
    # Create set of existing assessment keys
    existing_skill_keys = set()
    assessments_to_keep = []
    
    for assessment in existing_assessments:
        if not isinstance(assessment, dict):
            continue
            
        existing_skill = assessment.get('skill', '').strip()
        existing_category = assessment.get('category', '').strip()
        
        if not existing_skill or not existing_category:
            continue
            
        key = create_skill_key(existing_skill, existing_category)
        existing_skill_keys.add(key)
        
        # Keep assessment if skill still exists
        if key in current_skill_keys:
            assessments_to_keep.append(assessment)
            print(f"Keeping assessment for {existing_skill} ({existing_category})")
        else:
            print(f"Removing assessment for {existing_skill} ({existing_category})")
    
    # Find skills that need new assessments
    skills_needing_assessments = current_skill_keys - existing_skill_keys
    
    print(f"Assessments to keep: {len(assessments_to_keep)}")
    print(f"Skills needing new assessments: {len(skills_needing_assessments)}")
    
    # Generate assessments for missing skills
    successful_generations = 0
    
    if skills_needing_assessments:
        print("\nGenerating missing assessments...")
        
        for skill_key in skills_needing_assessments:
            skill_info = current_skills_map[skill_key]
            skill_name = skill_info["skill"]
            category = skill_info["category"]
            
            print(f"Generating assessment for {skill_name} ({category})")
            
            try:
                assessment = assessment_generator(skill_name, category)
                
                # Create new assessment object
                questions = assessment.get('questions', [])
                new_assessment_obj = {
                    "skill": skill_name,
                    "category": category,
                    "status": "not_started",
                    "score": "",
                    "totalQuestions": len(questions),
                    "attemptedQuestions": "",
                    "correctAnswers": "",
                    "questions": [],
                    "attemptedAt": "",
                    "completedAt": ""
                }
                
                # Process questions to match schema
                for question in questions:
                    question_obj = {
                        "questionId": question.get("questionId", ""),
                        "questionText": question.get("questionText", ""),
                        "options": question.get("options", []),
                        "correctOption": question.get("correctOption", ""),
                        "userAnswer": ""
                    }
                    new_assessment_obj["questions"].append(question_obj)
                
                assessments_to_keep.append(new_assessment_obj)
                successful_generations += 1
                print(f"Successfully generated assessment for {skill_name}")
                
            except Exception as e:
                print(f"Failed to generate assessment for {skill_name}: {str(e)}")
    
    # Update the user document with the synchronized assessments
    if db:
        try:
            user_ref = db.collection('user').document(user_id)
            user_ref.update({"assessment": assessments_to_keep})
            print(f"Updated user assessments. Total assessments: {len(assessments_to_keep)}")
        except Exception as e:
            raise Exception(f"Failed to update assessments in Firestore: {str(e)}")
    
    # Return summary statistics
    assessments_removed = len(existing_assessments) - len(assessments_to_keep) + successful_generations
    return {
        "assessments_generated": successful_generations,
        "assessments_kept": len(assessments_to_keep) - successful_generations,
        "assessments_removed": max(0, assessments_removed),  # Ensure non-negative
        "total_assessments": len(assessments_to_keep)
    }

def update_user_data(user_id: str, goal: str, roadmap_data: dict) -> bool:
    """Updates the user's document in Firestore with the generated goal and roadmap."""
    if not db:
        raise Exception("Firebase not initialized")
    
    try:
        user_doc_ref = db.collection('user').document(user_id)
        
        # Validate roadmap_data before proceeding
        if not roadmap_data:
            raise Exception("Invalid roadmap data provided")
        
        # Get existing document
        existing_doc = user_doc_ref.get()
        
        if existing_doc.exists:
            user_data = existing_doc.to_dict()
            existing_roadmaps = user_data.get('roadmap', [])
            
            # Ensure existing_roadmaps is a list and filter out any invalid entries
            if not isinstance(existing_roadmaps, list):
                existing_roadmaps = []
            else:
                # Remove any blank or invalid roadmaps
                existing_roadmaps = [
                    rm for rm in existing_roadmaps 
                    if rm and isinstance(rm, dict) and rm.get('title') and rm.get('tasks')
                ]
        else:
            existing_roadmaps = []
        
        # Add new roadmap to the array
        existing_roadmaps.append(roadmap_data)
        
        # Update the document
        user_doc_ref.update({
            'generated_goal': goal,
            'roadmap': existing_roadmaps
        })
        
        print(f"Successfully updated user document '{user_id}' with the generated goal and roadmap.")
        print(f"Total valid roadmaps for user: {len(existing_roadmaps)}")
        return True
        
    except Exception as e:
        raise Exception(f"Error updating Firestore: {str(e)}")

def process_assessments(user_id: str) -> Dict[str, Any]:
    """Main function to process assessments for a user"""
    try:
        if not db:
            raise Exception("Firebase not initialized")
        
        if not user_id or not user_id.strip():
            raise Exception("user_id cannot be empty")
        
        print(f"Processing assessments for user_id: {user_id}")
        
        # Sync assessments with current skills
        stats = sync_assessments_with_skills(user_id.strip())
        
        ai_mode = "LangChain" if LANGCHAIN_AVAILABLE else "Fallback"
        
        return {
            "success": True,
            "message": f"Assessments synchronized successfully using {ai_mode}",
            "user_id": user_id,
            **stats
        }
        
    except Exception as e:
        print(f"Error processing assessments for user {user_id}: {str(e)}")
        return {
            "success": False,
            "message": "Failed to process assessments",
            "user_id": user_id,
            "error_details": str(e)
        }

def process_user_id(user_id: str) -> Dict[str, Any]:
    """Main processing function that generates goal and roadmap for a user."""
    try:
        print(f"\nProcessing user_id: {user_id}")
        
        # Retrieve user skills
        print("Step 1: Retrieving user skills...")
        skills_data = get_user_skills(user_id)
        
        if not skills_data:
            raise Exception("No skills data found for user")
        
        print(f"Skills retrieved: {skills_data}")
        
        # Generate career goal
        print("Step 2: Generating career goal...")
        generated_goal = goal_generator(skills_data)
        
        # Generate roadmap
        print("Step 3: Generating roadmap...")
        roadmap = roadmap_generator(skills_data, generated_goal)
        
        # Update Firestore
        print("Step 4: Updating Firestore...")
        update_success = update_user_data(user_id, generated_goal, roadmap)
        
        if update_success:
            ai_mode = "LangChain" if LANGCHAIN_AVAILABLE else "Fallback"
            return {
                "success": True,
                "message": f"Goal and roadmap generated successfully using {ai_mode}",
                "user_id": user_id,
                "generated_goal": generated_goal,
                "roadmap": roadmap
            }
        else:
            raise Exception("Failed to update Firestore")
            
    except Exception as e:
        print(f"Error processing user {user_id}: {str(e)}")
        return {
            "success": False,
            "message": "Failed to process user",
            "user_id": user_id,
            "error_details": str(e)
        }

@app.post("/process-assessments", response_model=AssessmentResponse)
async def process_user_assessments(request: UserIdRequest):
    """Endpoint to sync assessments with user skills"""
    try:
        if not firebase_initialized:
            raise HTTPException(status_code=500, detail="Firebase initialization failed")
        
        if not request.user_id or not request.user_id.strip():
            raise HTTPException(status_code=400, detail="user_id cannot be empty")
        
        print(f"Received request to process assessments for user_id: {request.user_id}")
        
        # Process assessments
        result = process_assessments(request.user_id.strip())
        
        if result["success"]:
            return AssessmentResponse(**result)
        else:
            raise HTTPException(status_code=500, detail=result["error_details"])
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in assessments endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.post("/process-user", response_model=ProcessingResponse)
async def process_user(request: UserIdRequest):
    """Endpoint to process a user_id and generate goal/roadmap"""
    try:
        if not firebase_initialized:
            raise HTTPException(status_code=500, detail="Firebase initialization failed")
        
        if not request.user_id or not request.user_id.strip():
            raise HTTPException(status_code=400, detail="user_id cannot be empty")
        
        print(f"Received request to process user_id: {request.user_id}")
        
        # Process the user
        result = process_user_id(request.user_id.strip())
        
        if result["success"]:
            return ProcessingResponse(**result)
        else:
            raise HTTPException(status_code=500, detail=result["error_details"])
    
    except HTTPException:
        raise
    except Exception as e:
        print(f"Unexpected error in endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "firebase_initialized": firebase_initialized,
        "langchain_available": LANGCHAIN_AVAILABLE,
        "ai_mode": "LangChain" if LANGCHAIN_AVAILABLE else "Fallback",
        "message": "Service is running",
        "available_endpoints": [
            "POST /process-user - Generate goal and roadmap",
            "POST /process-assessments - Sync assessments with skills",
            "POST /get-userid - Legacy endpoint",
            "GET /health - Health check",
            "GET /debug-firebase - Debug Firebase connection"
        ]
    }

@app.get("/debug-firebase")
async def debug_firebase():
    """Debug Firebase initialization issues"""
    try:
        # Check environment variables
        firebase_key = os.environ.get('FIREBASE_SERVICE_ACCOUNT_KEY')
        google_creds = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS')
        google_project = os.environ.get('GOOGLE_CLOUD_PROJECT')
        
        debug_info = {
            "firebase_initialized": firebase_initialized,
            "db_object_exists": db is not None,
            "langchain_available": LANGCHAIN_AVAILABLE,
            "environment_variables": {
                "FIREBASE_SERVICE_ACCOUNT_KEY_exists": bool(firebase_key),
                "FIREBASE_SERVICE_ACCOUNT_KEY_length": len(firebase_key) if firebase_key else 0,
                "GOOGLE_APPLICATION_CREDENTIALS": google_creds,
                "GOOGLE_CLOUD_PROJECT": google_project,
            },
            "current_directory": os.getcwd(),
            "files_in_directory": os.listdir('.') if os.path.exists('.') else [],
        }
        
        # Check if modell directory exists
        if os.path.exists('modell'):
            debug_info["modell_directory_contents"] = os.listdir('modell')
        
        # Try to manually test Firebase initialization
        test_result = "not_attempted"
        try:
            if firebase_key:
                test_json = json.loads(firebase_key)
                test_result = f"JSON parse successful - project_id: {test_json.get('project_id', 'missing')}"
            else:
                test_result = "No Firebase key found in environment"
        except Exception as e:
            test_result = f"JSON parse failed: {str(e)}"
        
        debug_info["manual_test"] = test_result
        
        # Test database connection if Firebase is initialized
        if firebase_initialized and db:
            try:
                # Try a simple query
                test_ref = db.collection('user').limit(1)
                docs = list(test_ref.get())
                debug_info["database_test"] = f"✅ Database connection successful - found {len(docs)} documents"
            except Exception as e:
                debug_info["database_test"] = f"❌ Database connection failed: {str(e)}"
        else:
            debug_info["database_test"] = "❌ Cannot test - Firebase not initialized"
        
        return debug_info
        
    except Exception as e:
        return {
            "error": str(e),
            "firebase_initialized": firebase_initialized,
            "langchain_available": LANGCHAIN_AVAILABLE
        }

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "FastAPI service is running on Render"}

# Legacy endpoint for backward compatibility
@app.post("/get-userid")
async def get_userid(request: UserIdRequest):
    """Legacy endpoint - redirects to the new process-user endpoint"""
    return await process_user(request)

if _name_ == "_main_":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
updated roadmap code as well
