import sys
# If PlotNeuralNet is in a subfolder, add it to your path:
# sys.path.append('path/to/PlotNeuralNet')
from pycore.tikzeng import *
from pycore.blocks  import *

def my_arch():
    # 1. Define the overall architecture stack
    arch = [ 
        to_Head( '..', project_name='UBC-CMPE-401-LSTM-Modeling' ),
        to_Cor(), # Define the basic coordinate system
        to_Begin(), # Start the TikZ block
        
        # --- INPUT LAYER ---
        # Represented as a 3D block to show (Batch, 500, 1) volume
        to_Conv("input", 500, 1, offset="(0,0,0)", to="(0,0,0)", height=32, depth=32, width=1, caption="Input (Batch, 500, 1)"),

        # --- THE TRANSFORMER STACK (4x Blocks) ---
        # We represent this as a single large, labeled unit, then use PlotNeuralNet to "stack" blocks.
        # This approach is usually cleaner than defining 8 sequential layers (4 blocks * 2 sub-layers).
        to_Conv("encoder_1", 256, 1, offset="(3,0,0)", to="(input-east)", height=28, depth=28, width=15, caption="Transformer Block 1"),
        to_Conv("encoder_2", 256, 1, offset="(0,0,1)", to="(encoder_1-north)", height=28, depth=28, width=15, caption="Transformer Block 2"),
        to_Conv("encoder_3", 256, 1, offset="(0,0,1)", to="(encoder_2-north)", height=28, depth=28, width=15, caption="Transformer Block 3"),
        to_Conv("encoder_4", 256, 1, offset="(0,0,1)", to="(encoder_3-north)", height=28, depth=28, width=15, caption="Transformer Block 4"),
        
        # Draw a big bounding box around the 4 blocks to show they constitute the 4x stack.
        # PlotNeuralNet uses special macros for this, which are easiest to define via raw TikZ:
        to_node("transformer_label", r"\textbf{Pre-LN Transformer Stack}", offset="(0,1.5,0)", to="(encoder_4-north)"),
        to_draw("(-1,-2,0) -- (8,-2,0) -- (8,6,0) -- (-1,6,0) -- cycle;"), # Drawing a manual border
        
        # --- GLOBAL AVERAGE POOLING ---
        to_Pool("gap", offset="(3,0,0)", to="(encoder_1-east)", height=10, depth=10, width=1, caption="GlobalAvgPooling1D"),
        
        # --- FINAL DENSE STACK ---
        # The Final Layer (Output) with connection from GAP
        to_Dense("dense_128", 128, offset="(3,0,0)", to="(gap-east)", height=15, depth=15, width=1, caption="Dense (128, relu)"),
        to_Dense("output", 2, offset="(3,0,0)", to="(dense_128-east)", height=2, depth=2, width=1, caption="Dense (2, softmax)"),

        # --- CONNECTIONS ---
        # Connect Input -> 4x Block Stack
        to_connection( "input", "encoder_1"),
        # Connect 4x Block Stack -> GAP
        to_connection( "encoder_1", "gap"),
        # Connect GAP -> Dense 128
        to_connection( "gap", "dense_128"),
        # Connect Dense 128 -> Output
        to_connection( "dense_128", "output"),

        to_End() # End the TikZ block
    ]
    return arch

if __name__ == "__main__":
    name_file = "classification_transformer"
    arch = my_arch()
    # Generate the .tex file and compile to PDF (which can be converted to SVG)
    to_generate(arch, name_file + '.tex')