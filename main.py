import os
import json
import firebase_admin
from firebase_admin import credentials, firestore

from langchain_google_vertexai import ChatVertexAI
from langchain_core.prompts import ChatPromptTemplate

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, Any

app = FastAPI()

# Add CORS middleware with flexible origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for now - you can restrict later
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
    generated_goal: str = None
    roadmap: Dict[str, Any] = None
    error_details: str = None

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
        
        # Try different initialization methods
        # Method 1: Environment variable with JSON content
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
        local_path = "pacific-vault-470814-s5-25cf6513ecef.json"
        if os.path.exists(local_path):
            cred = credentials.Certificate(local_path)
            firebase_admin.initialize_app(cred)
            print("Firebase initialized with local credentials")
            return True
        
        print("All Firebase initialization methods failed")
        return False
        
    except Exception as e:
        print(f"Error initializing Firebase: {e}")
        return False

# Initialize Firebase when the module loads
firebase_initialized = initialize_firebase()
if firebase_initialized:
    db = firestore.client()
else:
    db = None

def get_user_skills(user_id: str) -> Dict[str, Any]:
    """Retrieves the skills map from a user document."""
    if not db:
        raise Exception("Firebase not initialized")
    
    try:
        user_doc_ref = db.collection('user').document(user_id)
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

            return skills_data
        else:
            raise Exception(f"No user document found with ID '{user_id}'")

    except Exception as e:
        raise Exception(f"Error retrieving user skills: {str(e)}")

def goal_generator(skills_data: dict) -> str:
    """Generates a career goal based on user's skills."""
    
    if not skills_data:
        raise Exception("Skills data is empty or invalid")

    try:
        # Use environment variable for project ID if available
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "pacific-vault-470814-s5")
        
        llm = ChatVertexAI(
            model="gemini-2.0-flash-exp", 
            temperature=0.3, 
            max_output_tokens=256, 
            project=project_id
        )

        # Convert the skills dictionary into a human-readable string for the LLM
        primary_skills = skills_data.get('primary', [])
        secondary_skills = skills_data.get('secondary', [])
        
        formatted_skills = f"Primary Skills: {', '.join(primary_skills)}. Secondary Skills: {', '.join(secondary_skills)}."

        prompt = ChatPromptTemplate([
            ("system", 
            "You are an expert career advisor. Based on the user's skills, suggest a specific and realistic career goal. "
            "The goal should be achievable within 1-2 years and should leverage their existing skills while helping them grow. "
            "Provide only the career goal as a concise statement (e.g., 'Full Stack Web Developer', 'Data Analyst specializing in Business Intelligence', etc.). "
            "Do not include explanations, just the career goal title."),
            ("user", "Based on my skills: {skills}, what career goal should I pursue?"),
        ])

        chain = prompt | llm
        response = chain.invoke({"skills": formatted_skills})
        career_goal = response.content.strip().strip('"\'')
        print(f"Generated Career Goal: {career_goal}")
        return career_goal
    except Exception as e:
        raise Exception(f"Failed to generate career goal: {str(e)}")

def roadmap_generator(skills_data: dict, user_goal: str) -> dict:
    """Generates a career or skill roadmap in dictionary format based on user background and goals."""
    
    if not skills_data or not user_goal:
        raise Exception("Skills data or user goal is empty or invalid")

    try:
        # Use environment variable for project ID if available
        project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "pacific-vault-470814-s5")
        
        llm = ChatVertexAI(
            model="gemini-2.0-flash-exp", 
            temperature=0.2, 
            max_output_tokens=1024, 
            project=project_id
        )

        # Convert the skills dictionary into a human-readable string for the LLM
        primary_skills = skills_data.get('primary', [])
        secondary_skills = skills_data.get('secondary', [])
        
        formatted_skills = f"Primary Skills: {', '.join(primary_skills)}. Secondary Skills: {', '.join(secondary_skills)}."

        prompt = ChatPromptTemplate([
            ("system", 
            "You are an expert career and skill roadmap generator. "
            "Generate a concise roadmap with 3-5 key tasks based on the user's skills and their goal. "
            "Return ONLY a valid JSON object with NO additional text, explanations, or markdown formatting. "
            "Use this exact structure: "
            '{{"title": "string describing the roadmap", '
            '"tasks": [{{"status": "completed/current/upcoming", "taskName": "specific task name", "timeAllocation": "time estimate", "resources": [{{"title": "resource name", "link": "https://valid-url.com"}}]}}]}}. '
            'Ensure all JSON values are properly quoted strings. Make sure the JSON is complete and valid.'),
            ("user", "My skills: {skills}. Generate a roadmap to become: {goal}"),
        ])

        chain = prompt | llm
        response = chain.invoke({"skills": formatted_skills, "goal": user_goal})
        
        # More robust JSON extraction
        raw_content = response.content.strip()
        print(f"Raw LLM Response: {raw_content}")
        
        # Remove any markdown code blocks or extra formatting
        raw_content = raw_content.replace('```json', '').replace('```', '').strip()
        
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
        raise Exception(f"JSON parsing error: {str(e)}")
    except Exception as e:
        raise Exception(f"Error in roadmap generation: {str(e)}")

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
            return {
                "success": True,
                "message": "Goal and roadmap generated successfully",
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
        "message": "Service is running"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {"message": "FastAPI service is running on GCP"}

# Optional: Keep the old endpoint for backward compatibility
@app.post("/get-userid")
async def get_userid(request: UserIdRequest):
    """Legacy endpoint - redirects to the new process-user endpoint"""
    return await process_user(request)

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
