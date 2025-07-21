const input = document.getElementById('todo-input');
const addBtn = document.getElementById('add-btn');
const todoList = document.getElementById('todo-list');

// Add task
addBtn.addEventListener('click', addTask);
input.addEventListener('keypress', function(e) {
  if (e.key === 'Enter') addTask();
});

function addTask() {
  const task = input.value.trim();
  if (task === '') return;

  const li = document.createElement('li');
  li.className = 'todo-item';
  li.innerHTML = `
    <label class="checkbox-label">
      <input type="checkbox" class="task-checkbox">
      <span class="task-text">${task}</span>
    </label>
  `;

  // Checkbox event: toggle completed
  const checkbox = li.querySelector('.task-checkbox');
  const taskText = li.querySelector('.task-text');
  checkbox.addEventListener('change', function() {
    if (checkbox.checked) {
      li.classList.add('completed');
    } else {
      li.classList.remove('completed');
    }
  });

  todoList.appendChild(li);
  input.value = '';
}