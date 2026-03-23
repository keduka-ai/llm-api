#!/bin/bash

# # URL of the FastAPI server
# URL="http://localhost:8000/api"
# # model_id="phi3v128ki"

# # Test Text Processing
# echo "Testing Text Processing..."
# TEXT_PAYLOAD=$(cat <<EOF
# {
#     "prompt": "what is the answer for 1+1? Explain it."
# }
# EOF
# )
# curl -X POST "$URL/process-text/" -H "Content-Type: application/json" -d "$TEXT_PAYLOAD"
# echo -e "\n"




#######################################################################################################################################

# # URL of the FastAPI server
# URL="http://localhost:8000/api"

# # Test Text Processing
# echo "\nTesting Text Processing..."
# TEXT_PAYLOAD=$(cat <<EOF
# {
#     "prompt": "Solve for \\\\( x \\\\) in the following: \\\\[ \\\\frac{x^2 - 3x + 2}{x^2 - 4} = \\\\frac{x - 1}{x + 2} \\\\]."
# }
# EOF
# )
# curl -X POST "$URL/text-prompt/" -H "Content-Type: application/json" -d "$TEXT_PAYLOAD"
# echo "\n"


#######################################################################################################################################

# URL of the FastAPI server
URL="http://localhost:8000/api"

# Test Text Processing
echo "\nTesting Text Processing..."
TEXT_PAYLOAD=$(cat <<EOF
{
    "system_prompt" : "You are a highly knowledgeable assistant specializing in providing detailed and professional explanations for math problems.",
    "prompt": "Solve the first-order differential equation: \\\\[ \\\\frac{dy}{dx} = 3y \\\\]"
}

EOF
)
curl -X POST "$URL/text-prompt/" -H "Content-Type: application/json" -d "$TEXT_PAYLOAD"
echo "\n"


#######################################################################################################################################

# URL of the FastAPI server
# URL="http://localhost:8000/api"

# # Test Text Processing
# echo "\nTesting Text Processing..."
# TEXT_PAYLOAD=$(cat <<EOF
# {
#     "system_prompt" : "You are a highly knowledgeable web development assistant specializing html, css, js, python and C++. You help humans with their web development problems. Use LaTex for equations and scientific notation. ",
    
#     "prompt": "convert the following to html: \n\n\nsolve the first-order differential equation \\\\(\\\\frac{dy}{dx} = 3y\\\\), we can use the method of separation of variables.\\n\\nFirst, rewrite the equation as:\\n\\n\\\\[\\\\frac{1}{y} dy = 3 dx\\\\]\\n\\nNow, integrate both sides with respect to their respective variables:\\n\\n\\\\[\\\\int \\\\frac{1}{y} dy = \\\\int 3 dx\\\\]\\n\\nThe left side integrates to \\\\(\\\\ln|y|\\\\), and the right side integrates to \\\\(3x + C\\\\), where \\\\(C\\\\) is the constant of integration:\\n\\n\\\\[\\\\ln|y| = 3x + C\\\\]\\n\\nTo solve for \\\\(y\\\\), we can exponentiate both sides:\\n\\n\\\\[e^{\\\\ln|y|} = e^{3x + C}\\\\]\\n\\nSince \\\\(e^{\\\\ln|y|} = |y|\\\\) and \\\\(e^{3x + C} = e^{3x}e^C = Ce^{3x}\\\\), we have:\\n\\n\\\\[|y| = Ce^{3x}\\\\]\\n\\nNow, we can remove the absolute value by considering two cases:\\n\\n1) \\\\(y = Ce^{3x}\\\\)\\n\\n2) \\\\(y = -Ce^{3x}\\\\)\\n\\nSo, the general solution to the first-order differential equation \\\\(\\\\frac{dy}{dx} = 3y\\\\) is:\\n\\n\\\\[y(x) = Ce^{3x}\\\\] or \\\\[y(x) = -Ce^{3x}\\\\]\\n\\nwhere \\\\(C\\\\) is an arbitrary constant."
# }
# EOF
# )
# curl -X POST "$URL/text-prompt/" -H "Content-Type: application/json" -d "$TEXT_PAYLOAD" 
# echo "\n"


