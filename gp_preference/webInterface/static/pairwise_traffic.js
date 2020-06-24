if (winner != -1){
    var items = document.getElementsByClassName('single-item');
    items[winner-1].classList.add('moving-copy');
    setTimeout(function(){ items[winner-1].classList.remove('moving-copy'); }, 400);
}

function target_el(event) {
    // depends on the browser what we have to call
    var target_el = null;
    if (typeof event.srcElement !== 'undefined') {
        target_el = event.srcElement;
    }
    if (typeof event.target !== 'undefined') {
        target_el = event.target;
    }
    return target_el;
}


function mouseEnterJob(event) {
    target_el(event).classList.add('pairwise-hovered-over');
}


function mouseLeaveJob(event) {
    target_el(event).classList.remove('pairwise-hovered-over');
    target_el(event).classList.remove('moving-copy');
}


function mouseClickJob(event) {

    // get the thing that was clicked
    var target = target_el(event);
    if (target.classList.contains('single-item-text')) {
        target = target.parentNode;
    }

    // get the item that was not clicked
    if (target.getAttribute('ID') == 'left') {
        var not_clicked = 'right';
        var target_not = document.getElementById('right');
    } else {
        var not_clicked = 'left';
        var target_not = document.getElementById('left');
    }

    // change style of clicked and non-clicked elements
    target.classList.add('just-dropped');
    target_not.classList.remove('just-dropped');

    // make buttons clickable
    var btns = document.getElementsByClassName('submit-button');
    for (var i; i<btns.length; i++) {
        btns[i].classList.remove('disabled');
    }
}

// called when the user submits the ranking
function handleSubmit(button_type) {

    // check if an item was chosen
    chosen_item = document.getElementsByClassName('just-dropped');
    if (chosen_item.length == 0) {
        window.alert('Please choose an item first.');
    }

    // add the button information to the element
    var btn_info = document.getElementsByName('buttonType');
    for (var i=0; i<btn_info.length; i++){
        btn_info[i].value = button_type;
    }

    // find the element that's currently clicked
    clicked = document.getElementsByClassName('just-dropped')[0];
    // get the item that was not clicked
    if (clicked.getAttribute('ID') == 'left') {
        var not_clicked = document.getElementById('right');
    } else {
        var not_clicked = document.getElementById('left');
    }
    // make not-clicked one disappear
    not_clicked.classList.add('invisible');

    // submit after a short time-out
    setTimeout(function(){ clicked.parentElement.submit(); }, 400);
}