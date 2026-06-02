#!/bin/bash

# List of compositions
compositions=("fe16cr00" "fe12cr04" "fe08cr08" "fe04cr12" "fe00cr16")

for comp in "${compositions[@]}"; do
    echo "========================================="
    echo "Running simulation for $comp"
    echo "========================================="
    
    # Go to composition directory
    cd "$comp"
    
    # Create composition-specific input file from template
    sed "s/COMPOSITION_NAME/$comp/g; s/MATERIAL_NAME/$comp/g" ../template.inp > "indentation_${comp}.inp"
    
    # Run CalculiX
    ccx "indentation_${comp}"
    
    # Check if successful
    if [ $? -eq 0 ]; then
        echo "✓ $comp completed successfully"
    else
        echo "✗ $comp failed - check log files"
    fi
    
    # Go back
    cd ..
    
    echo ""
done

echo "All simulations complete!"
