import re
import csv
import json
import openai

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


def generate_embedding(prompt):
    try:
        response = openai.Embedding.create(
            engine="text-embedding-3-large",
            input=prompt
        )
        return response['data'][0]['embedding']
    except Exception as e:
        print("Error:", str(e))
        return None


def create_embeddings_csv(descriptions, output_file="car_descriptions_embeddings.csv"):
    """
    Create CSV file with car descriptions and their embeddings
    
    Args:
        descriptions (list): List of car descriptions
        output_file (str): Output CSV filename
    
    Returns:
        str: Path to created CSV file
    """
    try:
        with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['car_description', 'embedding']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            # Write header
            writer.writeheader()
            
            # Process each description
            for i, description in enumerate(descriptions[:50], 1):
                print(f"Processing description {i}/{len(descriptions)}")
                # promt_embedding = f'This is short description about vehicles {description}. Generate vector embedding so that it can be used for semantic search.'
                
                # Generate embedding
                embedding = generate_embedding(description)
                
                if embedding:
                    # Convert embedding list to JSON string for CSV storage
                    embedding_json = json.dumps(embedding)
                    
                    # Write row
                    writer.writerow({
                        'car_description': description,
                        'embedding': embedding_json
                    })
                else:
                    print(f"Failed to generate embedding for description {i}")
                    # Write row with empty embedding
                    writer.writerow({
                        'car_description': description,
                        'embedding': ''
                    })
        
        print(f"CSV file created successfully: {output_file}")
        return output_file
        
    except Exception as e:
        print(f"Error creating CSV file: {str(e)}")
        return None

if __name__ == "__main__":
    # Get car descriptions
    descriptions = get_all_descriptions()
    
    # Create CSV with embeddings
    if descriptions:
        print(f"Creating embeddings CSV for {len(descriptions)} descriptions...")
        csv_file = create_embeddings_csv(descriptions)
        if csv_file:
            print(f"Embeddings CSV created: {csv_file}")
    else:
        print("No descriptions found")
