import re

def extract_descriptions_from_file(file_path):
    """
    Extract all car descriptions from the NLP text file and return as a list
    
    Args:
        file_path (str): Path to the car_details_nlp.txt file
    
    Returns:
        list: List of car descriptions
    """
    descriptions = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()
            
            # Split content by car entries (each starts with "Car #")
            car_entries = re.split(r'Car #\d+:', content)
            
            # Skip the first element (header content before first car)
            for entry in car_entries[1:]:
                if entry.strip():
                    # Extract just the description part (second line after car name)
                    lines = entry.strip().split('\n')
                    if len(lines) >= 2:
                        # First line is car name, second line is description
                        description = lines[1].strip()
                        if description:
                            descriptions.append(description)
    
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return []
    except Exception as e:
        print(f"Error reading file: {str(e)}")
        return []
    
    return descriptions

def get_all_descriptions():
    """
    Get all descriptions from the default car_details_nlp.txt file
    
    Returns:
        list: List of all car descriptions
    """
    return extract_descriptions_from_file("car_details_nlp.txt")

def print_descriptions(descriptions):
    """
    Print all descriptions with numbering
    
    Args:
        descriptions (list): List of descriptions to print
    """
    if not descriptions:
        print("No descriptions found.")
        return
    
    print(f"Found {len(descriptions)} car descriptions:\n")
    for i, desc in enumerate(descriptions, 1):
        print(f"{i}. {desc}\n")

#chunked data
descriptions = get_all_descriptions()
# print(descriptions[0])

def generate_response(prompt):
    try:
        response = openai.ChatCompletion.create(
            engine=config.DEPLOYMENT_NAME,
            messages = [
                {"role": "system", "content": "You are a helpful assistant that answers based on prompt"},
                {"role": "user", "content": prompt}],
            max_tokens=8000,
            temperature=0,
            )
    except Exception as e:
        print("Error:", str(e))
 
    return response['choices'][0]['message']['content']

promt_embedding = f'This is short description about vehicles {car_description}. Generate vector embedding so that it can be used for semantic search.'

for car_description in descriptions:
    response = generate_response(promt_embedding)
 