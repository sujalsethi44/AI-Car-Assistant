import csv
import os

def convert_csv_to_nlp(csv_file_path, output_file_path):
    """
    Convert CAR-DETAILS.csv to natural language descriptions
    """
    nlp_descriptions = []
    
    try:
        with open(csv_file_path, 'r', encoding='utf-8') as file:
            csv_reader = csv.DictReader(file)
            
            for row_num, row in enumerate(csv_reader, 1):
                # Extract data from each row
                name = row['name'].strip()
                year = row['year'].strip()
                selling_price = row['selling_price'].strip()
                km_driven = row['km_driven'].strip()
                fuel = row['fuel'].strip()
                seller_type = row['seller_type'].strip()
                transmission = row['transmission'].strip()
                owner = row['owner'].strip()
                
                # Format price in a readable way
                try:
                    price_num = int(selling_price)
                    if price_num >= 100000:
                        price_formatted = f"₹{price_num/100000:.1f} lakh"
                    else:
                        price_formatted = f"₹{price_num:,}"
                except:
                    price_formatted = f"₹{selling_price}"
                
                # Format kilometers
                try:
                    km_num = int(km_driven)
                    km_formatted = f"{km_num:,} km"
                except:
                    km_formatted = f"{km_driven} km"
                
                # Create natural language description with all details in one paragraph
                description = f"""Car #{row_num}: {name}
This {year} model {name} is a {fuel.lower()}-powered vehicle with {transmission.lower()} transmission, being sold by an {seller_type.lower()} seller for {price_formatted}. The car has been driven for {km_formatted} and is currently under {owner.lower()} ownership. This vehicle represents an excellent option for buyers seeking a reliable {fuel.lower()} car with {transmission.lower()} transmission in the used car market.

"""
                
                nlp_descriptions.append(description)
        
        # Write all descriptions to output file
        with open(output_file_path, 'w', encoding='utf-8') as output_file:
            output_file.write("CAR DETAILS - NATURAL LANGUAGE DESCRIPTIONS\n")
            output_file.write("=" * 50 + "\n\n")
            output_file.write(f"Total Cars: {len(nlp_descriptions)}\n")
            output_file.write("Generated from CAR-DETAILS.csv\n\n")
            output_file.write("=" * 50 + "\n\n")
            
            for description in nlp_descriptions:
                output_file.write(description + "\n")
        
        print(f"Successfully converted {len(nlp_descriptions)} car records to NLP format")
        print(f"Output saved to: {output_file_path}")
        
    except Exception as e:
        print(f"Error processing file: {str(e)}")

if __name__ == "__main__":
    csv_file = "CAR-DETAILS.csv"
    output_file = "car_details_nlp.txt"
    
    if os.path.exists(csv_file):
        convert_csv_to_nlp(csv_file, output_file)
    else:
        print(f"CSV file '{csv_file}' not found in current directory")
