# modil

# --- Core Imports ---
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from IPython.display import clear_output

# --- Data Loading & Preprocessing ---
# Load datasets
users = pd.read_csv('users.csv')
meals = pd.read_csv('meal_data(new).csv')

# Handle missing values
meals.fillna({'Common_Allergens': 'None', 'Diet_Preference': 'None'}, inplace=True)
users.fillna({'Allergens': 'None', 'Diet_Preference': 'None', 'Health_Restriction': 'None'}, inplace=True)

# Create meal tags for TF-IDF
meals['tags'] = meals['Diet_Preference'] + ' ' + meals['food_category'] + ' ' + meals['meal_types']

# --- Recommendation Engine Setup ---
tfidf = TfidfVectorizer(stop_words='english', lowercase=True)
tfidf_matrix = tfidf.fit_transform(meals['tags'])

def recommend_meals(user_id, n=5):
    """Core recommendation logic with safety filters"""
    try:
        user = users[users['UserID'] == user_id].iloc[0]
    except IndexError:
        return pd.DataFrame()

    filtered = meals.copy()
    
    # Allergen filtering with regex boundaries
    user_allergens = [a.strip().lower() for a in user['Allergens'].split(', ') if a.strip().lower() != 'none']
    if user_allergens:
        pattern = r'\b(' + '|'.join(re.escape(allergen) for allergen in user_allergens) + r')\b'
        filtered = filtered[~filtered['Common_Allergens'].str.lower().str.contains(
            pattern, na=False, regex=True
        )]
    
    # Health restrictions
    health_map = {'Heart': 'Heart_Healthy', 'Diabetic': 'Diabetic_Friendly'}
    if user['Health_Restriction'] != 'None':
        health_col = health_map.get(user['Health_Restriction'], user['Health_Restriction'])
        filtered = filtered[filtered[health_col] == 1]
    
    # Diet preferences
    user_diets = [d.strip().lower() for d in str(user['Diet_Preference']).split(',') if d.strip().lower() != 'none']
    if user_diets:
        filtered = filtered[filtered['Diet_Preference'].str.lower().str.contains(
            '|'.join(user_diets), na=False, regex=False
        )]
    
    if filtered.empty:
        return filtered
    
    # Ranking system
    if user_diets:
        user_profile = ' '.join(user_diets)
        user_vector = tfidf.transform([user_profile])
        cos_sim = cosine_similarity(user_vector, tfidf_matrix[filtered.index])
        filtered['similarity'] = cos_sim[0]
    else:
        filtered['similarity'] = 0  # Default for non-diet users
        
    return filtered.sort_values(
        by=['similarity', 'Protein (g)'], 
        ascending=[False, False]
    ).head(n)

# --- Interactive Recommendation System ---
class MealRecommender:
    def __init__(self):
        self.sessions = {}
        
    def _get_full_recommendations(self, user_id):
        """Get complete sorted meal list for user"""
        recs = recommend_meals(user_id, n=len(meals))
        return recs.reset_index(drop=True)
    
    def start_session(self, user_id):
        """Initialize new recommendation session"""
        full_recs = self._get_full_recommendations(user_id)
        self.sessions[user_id] = {
            'all_meals': full_recs,
            'current_pos': 0,
            'shown': []
        }
        return f"Session started for user {user_id} - {len(full_recs)} meals available"
    
    def get_meal(self, user_id, position=None):
        """Get specific meal from recommendations"""
        if user_id not in self.sessions:
            self.start_session(user_id)
            
        session = self.sessions[user_id]
        meals = session['all_meals']
        
        if meals.empty:
            return None, "No meals available"
        
        # Handle position requests
        if position is None:
            position = session['current_pos']
        else:
            position = max(0, min(position, len(meals)-1))
            
        meal = meals.iloc[position]
        session['current_pos'] = position
        
        # Track shown meals
        if position not in session['shown']:
            session['shown'].append(position)
            
        return meal, position+1  # Return 1-based index
    
    def next_meal(self, user_id, step=1):
        """Move through recommendation list"""
        session = self.sessions[user_id]
        new_pos = (session['current_pos'] + step) % len(session['all_meals'])
        return self.get_meal(user_id, new_pos)
    
    def interactive_session(self, user_id):
        """Command-line interface for meal browsing"""
        self.start_session(user_id)
        session = self.sessions[user_id]
        
        if session['all_meals'].empty:
            print("⚠️ No meals match your dietary requirements")
            return
            
        while True:
            clear_output(wait=True)
            meal, pos = self.get_meal(user_id)
            
            # Display current meal
            print(f" Recommendation #{pos}/{len(session['all_meals'])} ".center(40, '='))
            print(f"🍽 {meal['FoodName']}")
            print(f"🔹 Calories: {meal['Calories']}")
            print(f"🔹 Protein: {meal['Protein (g)']}g")
            print(f"🔹 Allergens: {meal['Common_Allergens'] or 'None'}")
            print(f"🔹 Diets: {meal['Diet_Preference']}")
            print(f"🔹 Similarity Score: {meal.get('similarity', 0):.2f}")
            
            # Navigation controls
            print("\nControls:")
            print("1. Next meal ➡")
            print("2. Previous meal ⬅")
            print("3. Random new meal 🎲")
            print("4. Restart session 🔄")
            print("5. Exit ❌")
            
            choice = input("\nYour choice (1-5): ").strip()
            
            if choice == '1':
                self.next_meal(user_id, 1)
            elif choice == '2':
                self.next_meal(user_id, -1)
            elif choice == '3':
                self.next_meal(user_id, len(session['all_meals'])//2)  # Example random jump
            elif choice == '4':
                self.start_session(user_id)
            elif choice == '5':
                print("👋 Ending recommendation session")
                break
            else:
                print("⚠️ Invalid input, please try again")

# --- Usage Example ---
if __name__ == "__main__":
    # Initialize recommender system
    recommender = MealRecommender()
    
    # Start interactive session for user 16
    recommender.interactive_session(16)
