import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Get the absolute path of this script's directory (Plots/)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Get the absolute path of the project root directory (ReachAvoidAlgo/)
# Goes up one level from 'scripts/'
project_root_dir = os.path.dirname(current_dir)
# Add the project root directory to sys.path
sys.path.insert(0, project_root_dir)

import HelperFunctions

if __name__ == "__main__":
    try:
        # Load all parameters
        m, L, q_start, q_end, min_tau_loaded, max_tau_loaded = HelperFunctions.load_parameters_from_file('parameters.txt')

        print("\n--- Loaded Parameters ---")
        print("Masses (m):", m)
        print("Lengths (L):", L)
        print("Path Start (q_start_rad):", q_start)
        print("Path End (q_end_rad):", q_end)
        print("Min Torques (min_tau):", min_tau_loaded)
        print("Max Torques (max_tau):", max_tau_loaded)
        
        qA_1 = np.array([q_start[0], q_end[0]])
        qA_2 = np.array([q_start[1], q_end[1]+3])
        theta = np.array([3, 3])
        qA_ref1_1 = -qA_1
        qA_ref1_2 = qA_2
        qA_addTheta1 = qA_1 + theta
        qA_addTheta2 = qA_2
        qA_minusTheta1 = qA_1 - theta
        qA_minusTheta2 = qA_2
        qA_ref2_1 = qA_1
        qA_ref2_2 = -qA_2
        
        qAreverse_1 = qA_1 + theta/2
        qAreverse_2 = qA_2
        
        
        # Create plot
        fig, ax = plt.subplots(figsize=(8, 6))
        # Set axis labels (q1, q2)
        ax.set_xlabel('$q_1$', loc='right', fontsize=14)
        ax.set_ylabel('$q_2$', loc='top', rotation=0, fontsize=14)
        # Remove numbers from the axes
        ax.set_xticks([])
        ax.set_yticks([])
        # Move left spine to x=0
        ax.spines['left'].set_position('zero')
        # Move bottom spine to y=0
        ax.spines['bottom'].set_position('zero')
        # Remove right spine
        ax.spines['right'].set_color('none')
        # Remove top spine
        ax.spines['top'].set_color('none')
        # Add arrows to the end of the axes
        ax.plot(1, 0, ">k", transform=ax.get_yaxis_transform(), clip_on=False, markersize=8)
        ax.plot(0, 1, "^k", transform=ax.get_xaxis_transform(), clip_on=False, markersize=8)
        # Set axis limits
        ax.set_xlim(-4, 4)
        ax.set_ylim(-1, 6)
        ax.set_aspect('equal', adjustable='box')
        
        # Plot paths
        # Line color, Solid line, Circle markers at the pts, marker size, marker fill color, marker edge color, legend label
        ax.plot(qA_1, qA_2, color='red', linestyle='-', linewidth=1, marker='o', markersize=8, markerfacecolor='black', markeredgecolor='black')
        #ax.plot(qA_ref1_1, qA_ref1_2, color='blue', linestyle='--', linewidth=1, marker='o', markersize=8, markerfacecolor='black', markeredgecolor='black')
        # ax.plot(qA_addTheta1, qA_addTheta2, color='blue', linestyle='--', linewidth=1, marker='o', markersize=8, markerfacecolor='black', markeredgecolor='black')
        # ax.plot(qA_minusTheta1, qA_minusTheta2, color='orange', linestyle='--', linewidth=1, marker='o', markersize=8, markerfacecolor='black', markeredgecolor='black')
        
        shade = False
        # Labels
        # q2 max label
        if False:
            ax.set_yticks([qA_2[1]])
            # Hide label for y-axis i.e qA_2[1]
            ax.tick_params(axis='y', which='major', length=5, width=1.5, colors='black', labelleft=False)
            ax.text(0, qA_2[1], '$q_{2_{max}}$', fontsize=12, color= 'black', ha='right', va='bottom')
        # qA
        if True:
            ax.text(qA_1[0] + 0.3, qA_2[0] - 0.5, '$q_A(0)$', fontsize=12, color='tab:red', ha='left', va='bottom')
            ax.text(qA_1[1], qA_2[1] + 0.2, '$q_A(1)$', fontsize=12, color='tab:red', ha='right', va='bottom')
            ax.text(qA_1[1]/2 + 0.05, qA_2[1]/2 + 0.5, '$q_A(x_1)$', fontsize=12, color='tab:red', ha='center', va='center', rotation=55)
            
        # -qA
        if True:
            ax.text(qA_1[0] + 0.3, qA_2[0] - 0, '$q_A\'(1)$', fontsize=12, color='black', ha='left', va='bottom')
            ax.text(qA_1[1] + 0.2, qA_2[1] + 0.2, '$q_A\'(0)$', fontsize=12, color='black', ha='left', va='bottom')
            ax.text(qA_1[1]/2 + 0.6, qA_2[1]/2 + 0.2, '$q_A\' = q_A(1-x_1)$', fontsize=12, color='black', ha='center', va='center', rotation=55)
        #qA + theta
        if False:
            ax.text(qA_addTheta1[0] - 0.1, qA_addTheta2[0] - 0.1, r'$q_A(0) + \theta$', fontsize=12, color='blue', ha='left', va='top')
            ax.text(qA_addTheta1[1] + 0.1, qA_addTheta2[1] + 0.2, r'$q_A(1) + \theta$', fontsize=12, color='blue', ha='left', va='bottom')
            ax.text(qA_addTheta1[1]-1, qA_addTheta2[1]/2, r'$q_A(x_1) + \theta$', fontsize=12, color='blue', ha='center', va='center', rotation=55)
            shade = True
        #qA - theta
        if False:
            ax.text(qA_minusTheta1[0] - 0.1, qA_minusTheta2[0] - 0.1, r'$q_A(0) - \theta$', fontsize=12, color='orange', ha='left', va='top')
            ax.text(qA_minusTheta1[1] + 0.1, qA_minusTheta2[1] + 0.2, r'$q_A(1) - \theta$', fontsize=12, color='orange', ha='left', va='bottom')
            ax.text(qA_minusTheta1[1]-1, qA_minusTheta2[1]/2, r'$q_A(x_1) - \theta$', fontsize=12, color='orange', ha='center', va='center', rotation=55)
            shade = True
        # qA reflected q1
        if False:
            ax.text(qA_ref1_1[0] - 0.1, qA_ref1_2[0] -0.1, '$q_A\'\'(0)$', fontsize=12, color='black', ha='right', va='top')
            ax.text(qA_ref1_1[1] + 0.1, qA_ref1_2[1] + 0.2, '$q_A\'\'(1)$', fontsize=12, color='black', ha='left', va='bottom')
            ax.text(qA_ref1_1[1]/2 + 0.5, qA_ref1_2[1]/2, '$q_A\'\'(x_1)$', fontsize=12, color='black', ha='center', va='center', rotation=-55)
            
        # Shade reachable set
        if shade:
            # Shade between the min and max q2 lines
            ax.axhspan(qA_2[0], qA_2[1], color='lightgray', alpha=0.5)
            # Plot dashed lines for reachable set
            ax.axhline(qA_2[1], color='gray', linestyle='--', linewidth=1)
            ax.axhline(qA_2[0], color='gray', linestyle='--', linewidth=1)
        
        # Set aspect ratio for equal scaling (optional, but often good for geometry)
        ax.set_aspect('equal', adjustable='box')
        # Display the figure
        plt.show()
    
    except (FileNotFoundError, ValueError) as e:
        print(f"Exiting due to parameter loading or processing error: {e}")