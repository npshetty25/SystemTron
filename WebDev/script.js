    let display = document.getElementById('display');
    let buttons = Array.from(document.querySelectorAll('.buttons button'));

    buttons.map(button => {
        button.addEventListener('click', (e) => {
            const val = e.target.innerText;
            if (val === 'AC') {
                display.value = '';
            } else if (val === '=') {
                try {
                    display.value = eval(display.value);
                } catch {
                    display.value = 'Error';
                }
            } else {
                display.value += val;
            }
        });
    });
