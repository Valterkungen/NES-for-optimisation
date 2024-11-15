import pandas as pd

# The input text
text = """
Running Gradient Descent...
GD Iteration 0 : Value = 48.4857, Parameters = [  0.753  -0.832]
GD Iteration 5 : Value = 169.0634, Parameters = [  2.000  -0.655]
GD Iteration 10 : Value = 165.2839, Parameters = [  2.000  2.000]
GD Iteration 15 : Value = 165.2839, Parameters = [  2.000  2.000]
GD Iteration 20 : Value = 165.2839, Parameters = [  2.000  2.000]
GD Iteration 25 : Value = 165.2839, Parameters = [  2.000  2.000]
GD Iteration 30 : Value = 165.2839, Parameters = [  2.000  2.000]
GD Iteration 35 : Value = 165.2839, Parameters = [  2.000  2.000]
GD Iteration 40 : Value = 165.2839, Parameters = [  2.000  2.000]
GD Iteration 45 : Value = 165.2839, Parameters = [  2.000  2.000]
GD Iteration 50 : Value = 165.2839, Parameters = [  2.000  2.000]
GD Iteration 55 : Value = 165.2839, Parameters = [  2.000  2.000]
GD Iteration 60 : Value = 165.2839, Parameters = [  2.000  2.000]
GD Iteration 65 : Value = 165.2839, Parameters = [  2.000  2.000]
GD Iteration 70 : Value = 165.2839, Parameters = [  2.000  2.000]
GD Iteration 75 : Value = 165.2839, Parameters = [  2.000  2.000]
GD Iteration 80 : Value = 165.2839, Parameters = [  2.000  2.000]
GD Iteration 85 : Value = 165.2839, Parameters = [  2.000  2.000]
GD Iteration 90 : Value = 165.2839, Parameters = [  2.000  2.000]
GD Iteration 95 : Value = 165.2839, Parameters = [  2.000  2.000]

Running NES with population 50...
NES Generation 0 : Value = 96.5295, Parameters = [  1.139  1.932]
NES Generation 5 : Value = 63.8607, Parameters = [  0.879  1.913]
NES Generation 10 : Value = 28.9612, Parameters = [  0.642  1.899]
NES Generation 15 : Value = 5.4387, Parameters = [  0.488  1.879]
NES Generation 20 : Value = 4.5522, Parameters = [  0.468  1.874]
NES Generation 25 : Value = 4.9204, Parameters = [  0.484  1.860]
NES Generation 30 : Value = 6.0026, Parameters = [  0.495  1.855]
NES Generation 35 : Value = 5.1796, Parameters = [  0.489  1.842]
NES Generation 40 : Value = 4.3063, Parameters = [  0.466  1.830]
NES Generation 45 : Value = 3.9170, Parameters = [  0.480  1.787]
NES Generation 50 : Value = 4.4243, Parameters = [  0.489  1.776]
NES Generation 55 : Value = 4.9529, Parameters = [  0.495  1.759]
NES Generation 60 : Value = 3.5418, Parameters = [  0.481  1.748]
NES Generation 65 : Value = 3.7541, Parameters = [  0.489  1.715]
NES Generation 70 : Value = 4.5792, Parameters = [  0.497  1.711]
NES Generation 75 : Value = 4.1188, Parameters = [  0.494  1.697]
NES Generation 80 : Value = 4.3294, Parameters = [  0.495  1.703]
NES Generation 85 : Value = 2.8087, Parameters = [  0.480  1.677]
NES Generation 90 : Value = 2.7102, Parameters = [  0.477  1.669]
NES Generation 95 : Value = 2.8528, Parameters = [  0.485  1.660]"""

data = []

# Process each line
for line in text.split('\n'):
    line = line.strip()
    
    # Skip empty lines and the "Running NES" line
    if not line or line.startswith('Running'):
        continue
        
    # Check if it's a GD line
    if line.startswith('GD'):
        # Split the line into components
        parts = line.replace('GD Iteration ', '').split(' : Value = ')
        iteration = int(parts[0])
        
        # Extract value and parameters
        value_params = parts[1].split(', Parameters = ')
        value = float(value_params[0])
        
        # Clean up parameters string and convert to floats
        params = value_params[1].strip('[]').split()
        param1 = float(params[0])
        param2 = float(params[1])
        
        data.append({
            'Type': 'GD',
            'Iteration': iteration,
            'Value': value,
            'Parameter1': param1,
            'Parameter2': param2
        })
        
    # Check if it's an NES line
    elif line.startswith('NES'):
        # Split the line into components
        parts = line.replace('NES Generation ', '').split(' : Value = ')
        iteration = int(parts[0])
        
        # Extract value and parameters
        value_params = parts[1].split(', Parameters = ')
        value = float(value_params[0])
        
        # Clean up parameters string and convert to floats
        params = value_params[1].strip('[]').split()
        param1 = float(params[0])
        param2 = float(params[1])
        
        data.append({
            'Type': 'NES',
            'Iteration': iteration,
            'Value': value,
            'Parameter1': param1,
            'Parameter2': param2
        })

# Create DataFrame
df = pd.DataFrame(data)

# Sort by Type and Iteration
df = df.sort_values(['Type', 'Iteration'])

# Print debugging information
print(f"Total lines processed: {len(data)}")
print(f"GD iterations: {len(df[df['Type'] == 'GD'])}")
print(f"NES generations: {len(df[df['Type'] == 'NES'])}")
print(f"\nDataFrame shape: {df.shape}")

# Print all iterations to verify
print("\nGD Iterations found:", sorted(df[df['Type'] == 'GD']['Iteration'].tolist()))
print("NES Iterations found:", sorted(df[df['Type'] == 'NES']['Iteration'].tolist()))

# Save to CSV
df.to_csv("output_ice.csv", index=False)

print("\nData saved to output_ice.csv")
