document.addEventListener('DOMContentLoaded', function() {
    // Elements
    const userMenuButton = document.getElementById('userMenuButton');
    const userDropdown = document.getElementById('userDropdown');
    const dropdownArrow = document.getElementById('dropdownArrow');
    const viewAnswersBtn = document.getElementById('viewAnswersBtn');
    const answersModal = document.getElementById('answersModal');
    const closeModal = document.getElementById('closeModal');
    const cancelChangesBtn = document.getElementById('cancelChangesBtn');
    const saveChangesBtn = document.getElementById('saveChangesBtn');
    const answersContainer = document.getElementById('answersContainer');

    // Toggle dropdown
    function toggleDropdown() {
        const isExpanded = userMenuButton.getAttribute('aria-expanded') === 'true';
        userMenuButton.setAttribute('aria-expanded', !isExpanded);
        userDropdown.classList.toggle('hidden');
        dropdownArrow.classList.toggle('rotate-180');
    }

    // Close dropdown
    function closeDropdown() {
        userMenuButton.setAttribute('aria-expanded', 'false');
        userDropdown.classList.add('hidden');
        dropdownArrow.classList.remove('rotate-180');
    }

    // Event Listeners
    if (userMenuButton && userDropdown) {
        // Toggle dropdown on button click
        userMenuButton.addEventListener('click', function(e) {
            e.stopPropagation();
            toggleDropdown();
        });

        // Close dropdown when clicking outside
        document.addEventListener('click', function(e) {
            if (!userMenuButton.contains(e.target) && !userDropdown.contains(e.target)) {
                closeDropdown();
            }
        });

        // Close on Escape key
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Escape' && !userDropdown.classList.contains('hidden')) {
                closeDropdown();
            }
        });
    }

    // View answers button click
    if (viewAnswersBtn) {
        viewAnswersBtn.addEventListener('click', function(e) {
            e.preventDefault();
            closeDropdown();
            if (answersModal) {
                answersModal.classList.remove('hidden');
                document.body.style.overflow = 'hidden'; // Prevent background scrolling
            }
            if (typeof fetchUserAnswers === 'function') {
                fetchUserAnswers();
            }
        });
    }

    // Close modal
    function closeAnswersModal() {
        answersModal.classList.add('hidden');
    }

    closeModal.addEventListener('click', closeAnswersModal);
    cancelChangesBtn.addEventListener('click', closeAnswersModal);

    // Save changes
    saveChangesBtn.addEventListener('click', function() {
        const formData = new FormData();
        const inputs = answersContainer.querySelectorAll('input, textarea, select');
        
        inputs.forEach(input => {
            if (input.type === 'checkbox' || input.type === 'radio') {
                if (input.checked) {
                    formData.append(input.name, input.value);
                }
            } else {
                formData.append(input.name, input.value);
            }
        });

        fetch('/update-answers', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.success) {
                alert('Your answers have been updated successfully!');
                closeAnswersModal();
            } else {
                alert('Error updating answers: ' + (data.error || 'Unknown error'));
            }
        })
        .catch(error => {
            console.error('Error:', error);
            alert('An error occurred while updating your answers.');
        });
    });

    // Fetch user answers
    function fetchUserAnswers() {
        fetch('/get-user-answers')
            .then(response => response.json())
            .then(data => {
                if (data.success) {
                    displayAnswers(data.answers);
                    answersModal.classList.remove('hidden');
                } else {
                    alert('Error loading your answers: ' + (data.error || 'No answers found'));
                }
            }) 
            .catch(error => {
                console.error('Error:', error);
                alert('An error occurred while loading your answers.');
            });
    }

    // Display answers in the modal
    function displayAnswers(answers) {
        answersContainer.innerHTML = '';
        
        if (!answers || Object.keys(answers).length === 0) {
            answersContainer.innerHTML = `
                <div class="text-center py-8">
                    <i class="fas fa-inbox text-4xl text-gray-300 mb-2"></i>
                    <p class="text-gray-700">You haven't submitted any answers yet.</p>
                </div>
            `;
            saveChangesBtn.style.display = 'none';
            return;
        }

        saveChangesBtn.style.display = 'inline-flex';
        
        // Get the questionnaire template to determine field types
        fetch('/questionnaire')
            .then(response => response.text())
            .then(html => {
                const parser = new DOMParser();
                const doc = parser.parseFromString(html, 'text/html');
                const form = doc.getElementById('questionnaireForm');
                
                if (!form) {
                    throw new Error('Could not find questionnaire form');
                }
                
                // Create a form to hold all the fields
                const formElement = document.createElement('form');
                formElement.className = 'space-y-6';
                
                for (const [question, answer] of Object.entries(answers)) {
                    if (answer === null || answer === '') continue;
                    
                    const answerDiv = document.createElement('div');
                    answerDiv.className = 'space-y-2 border-b border-gray-100 py-4';
                    
                    // Format the question (replace underscores with spaces and capitalize first letter)
                    const formattedQuestion = question
                        .replace(/_/g, ' ')
                        .replace(/\b\w/g, l => l.toUpperCase());
                    
                    // Find the corresponding question in the template
                    const questionElement = form.querySelector(`[name="${question}"]`);
                    
                    if (questionElement) {
                        const questionType = questionElement.type || 
                                          (questionElement.tagName === 'TEXTAREA' ? 'textarea' : 
                                           questionElement.tagName === 'SELECT' ? 'select' : 'text');
                        
                        const label = document.createElement('label');
                        label.className = 'block text-sm font-medium text-gray-700';
                        label.textContent = formattedQuestion;
                        answerDiv.appendChild(label);
                        
                        if (questionType === 'select') {
                            const select = document.createElement('select');
                            select.name = question;
                            select.className = 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-green-500 focus:ring focus:ring-green-200 focus:ring-opacity-50 text-gray-800';
                            
                            // Add default option
                            const defaultOption = document.createElement('option');
                            defaultOption.value = '';
                            defaultOption.textContent = '-- Select --';
                            select.appendChild(defaultOption);
                            
                            // Add options from the original select
                            Array.from(questionElement.options).forEach(option => {
                                if (option.value) {
                                    const newOption = document.createElement('option');
                                    newOption.value = option.value;
                                    newOption.textContent = option.textContent;
                                    if (answer === option.value || answer.toString() === option.value) {
                                        newOption.selected = true;
                                    }
                                    select.appendChild(newOption);
                                }
                            });
                            
                            answerDiv.appendChild(select);
                        } 
                        else if (questionType === 'radio') {
                            const radioContainer = document.createElement('div');
                            radioContainer.className = 'mt-2 space-y-2';
                            
                            // Get the radio buttons for this question
                            const radioButtons = form.querySelectorAll(`input[type="radio"][name="${question}"]`);
                            
                            radioButtons.forEach((radio, index) => {
                                const radioDiv = document.createElement('div');
                                radioDiv.className = 'flex items-center';
                                
                                const input = document.createElement('input');
                                input.type = 'radio';
                                input.id = `${question}_${index}`;
                                input.name = question;
                                input.value = radio.value;
                                input.className = 'h-4 w-4 text-green-600 focus:ring-green-500 border-gray-300';
                                if (answer === radio.value || answer.toString() === radio.value) {
                                    input.checked = true;
                                }
                                
                                const label = document.createElement('label');
                                label.htmlFor = `${question}_${index}`;
                                label.className = 'ml-3 block text-sm text-gray-700';
                                label.textContent = radio.nextElementSibling?.textContent || radio.value;
                                
                                radioDiv.appendChild(input);
                                radioDiv.appendChild(label);
                                radioContainer.appendChild(radioDiv);
                            });
                            
                            answerDiv.appendChild(radioContainer);
                        }
                        else if (questionType === 'textarea') {
                            const textarea = document.createElement('textarea');
                            textarea.name = question;
                            textarea.rows = 3;
                            textarea.className = 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-green-500 focus:ring focus:ring-green-200 focus:ring-opacity-50 text-gray-800';
                            textarea.textContent = answer;
                            answerDiv.appendChild(textarea);
                        }
                        else if (questionType === 'date') {
                            const input = document.createElement('input');
                            input.type = 'date';
                            input.name = question;
                            input.value = answer;
                            input.className = 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-green-500 focus:ring focus:ring-green-200 focus:ring-opacity-50 text-gray-800';
                            answerDiv.appendChild(input);
                        }
                        else {
                            // Fallback to text input
                            const input = document.createElement('input');
                            input.type = 'text';
                            input.name = question;
                            input.value = Array.isArray(answer) ? answer.join(', ') : answer;
                            input.className = 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-green-500 focus:ring focus:ring-green-200 focus:ring-opacity-50 text-gray-800';
                            answerDiv.appendChild(input);
                        }
                    } else {
                        // Fallback for questions not in the current template
                        const input = document.createElement('input');
                        input.type = 'text';
                        input.name = question;
                        input.value = Array.isArray(answer) ? answer.join(', ') : answer;
                        input.className = 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-green-500 focus:ring focus:ring-green-200 focus:ring-opacity-50 text-gray-800';
                        
                        const label = document.createElement('label');
                        label.className = 'block text-sm font-medium text-gray-700';
                        label.textContent = formattedQuestion;
                        
                        answerDiv.appendChild(label);
                        answerDiv.appendChild(input);
                    }
                    
                    formElement.appendChild(answerDiv);
                }
                
                // Add the form to the container
                answersContainer.appendChild(formElement);
            })
            .catch(error => {
                console.error('Error loading questionnaire template:', error);
                // Fallback to simple display if there's an error
                displaySimpleAnswers(answers);
            });
    }
           // Simple display of answers
           function displaySimpleAnswers(answers) {
               answersContainer.innerHTML = '';
               
               for (const [question, answer] of Object.entries(answers)) {
                   const answerDiv = document.createElement('div');
                   answerDiv.className = 'space-y-2 border-b border-gray-100 py-4';
                   
                   const label = document.createElement('label');
                   label.className = 'block text-sm font-medium text-gray-700';
                   label.textContent = question.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase());
                   
                   const input = document.createElement('input');
                   input.type = 'text';
                   input.name = question;
                   input.value = Array.isArray(answer) ? answer.join(', ') : answer;
                   input.className = 'mt-1 block w-full rounded-md border-gray-300 shadow-sm focus:border-green-500 focus:ring focus:ring-green-200 focus:ring-opacity-50 text-gray-800';
                   
                   answerDiv.appendChild(label);
                   answerDiv.appendChild(input);
                   answersContainer.appendChild(answerDiv);
               }
           }
});
