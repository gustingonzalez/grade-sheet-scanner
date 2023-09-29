/**
 * Adds a row in the students table.
 * @param {*} id
 * @param {*} name
 */
function addStudentRowInStudentsTable(id, name) {
  const tbody = document.querySelector("#students-table tbody");
  const row = tbody.insertRow(-1);

  row.insertCell(0).textContent = id;
  row.insertCell(1).textContent = name;

  const gradeCell = row.insertCell(2);
  const input = document.createElement("input");
  input.type = "number";
  input.min = 0;
  input.id = `grade-${id}`;
  input.name = "grade";
  input.placeholder = "Ausente";
  input.style.textAlign = "center";
  gradeCell.appendChild(input);
}

/**
 * Adds the header to the students table.
 */
function addHeaderInStudentsTable() {
  const tbody = document.querySelector("#students-table tbody");
  const header = tbody.insertRow(0);
  idCell = header.insertCell(0);
  idCell.textContent = "Legajo";
  idCell.style.visibility = "hidden";

  studentCell = header.insertCell(1);
  studentCell.textContent = "Estudiante";
  studentCell.style.visibility = "hidden";

  gradeCell = header.insertCell(2);
  gradeCell.textContent = "Calificación";
  gradeCell.style.textAlign = "center";
}

/**
 * Clears the student table.
 */
function clearStudentsTable() {
  const tbody = document.querySelector("#students-table tbody");
  tbody.innerHTML = "";
}

/**
 * Fills the student table with the provided grades.
 * @param {*} grades
 */
function fillGradesInStudentsTable(grades) {
  // Clears existing data.
  clearStudentsTable();

  // Adds a header.
  addHeaderInStudentsTable();

  // Adds rows.
  for (const student of students) {
    addStudentRowInStudentsTable(student.id, student.name);
  }

  // Gets the table body and its rows.
  const tbody = document.querySelector("#students-table tbody");
  const rows = tbody.getElementsByTagName("tr");

  // Populates grades for each student.
  for (let i = 1; i < rows.length; i++) {
    const grade = grades[i - 1];
    const inputElement = rows[i].querySelector('input[name="grade"]');

    if (inputElement && grade) {
      inputElement.value = grade;
    }
  }
}

/**
 * Shows/hides the elements marked as 'shown-after-scan' based on the 'show'
 * parameter.
 * @param {Boolean} show
 */
function showDisplayableElementsAfterScan(show) {
  const shownAfterScanElements = document.querySelectorAll(".shown-after-scan");
  shownAfterScanElements.forEach((element) => {
    element.style.display = show ? "initial" : "none";
  });
}

/**
 * Scans and processes an image.
 */
function scan_and_process_image() {
  showDisplayableElementsAfterScan(false);

  const instructions = document.querySelector("#instructions");
  instructions.style.display = "none";

  const scanErrorMessage = document.querySelector("#scan-error-message");
  scanErrorMessage.style.display = "none";

  const scanInProgressMessage = document.querySelector(
    "#scan-in-progress-message"
  );
  scanInProgressMessage.style.display = "block";

  const scanButton = document.querySelector("#scan-button");
  scanButton.style.display = "none";

  fetch("/scan_and_process_image", {
    method: "GET",
  })
    .then((response) => response.json())
    .then((data) => {
      fillGradesInStudentsTable(data);
      showDisplayableElementsAfterScan(true);
    })
    .catch((error) => {
      console.error(`Error scanning/processing the image: ${error}.`);
      scanInProgressMessage.style.display = "none";
      scanErrorMessage.style.display = "block";
      showDisplayableElementsAfterScan(false);
    })
    .finally(() => {
      scanInProgressMessage.style.display = "none";
      scanButton.style.display = "initial";
    });
}

/**
 * Downloads the specified csv.
 * @param {*} csv
 * @param {*} filename
 */
function downloadCSV(csv, filename) {
  const csvFile = new Blob([csv], { type: "text/csv" });

  // Creates a temporary download link for the downloading url.
  const downloadLink = document.createElement("a");
  downloadLink.download = filename;
  downloadLink.href = window.URL.createObjectURL(csvFile);
  downloadLink.style.display = "none";
  document.body.appendChild(downloadLink);

  // Performs the download by clicking the temporary download link.
  downloadLink.click();

  // Removes the temporary download link.
  document.body.removeChild(downloadLink);
}

/**
 * Exports the students (grade sheet) table to a csv file.
 */
function exportStudentsTableToCSV() {
  const table = document.querySelector("#students-table");
  let csv = [];

  for (let i = 0; i < table.rows.length; i++) {
    const row = table.rows[i];
    let rowData = [];

    for (let j = 0; j < row.cells.length; j++) {
      const cell = row.cells[j];

      // Handles the input of the last column.
      const input = cell.querySelector("input");
      if (input) {
        rowData.push(`"${input.value}"`);
      } else {
        // If there is no input, uses the cell's text content.
        rowData.push(`"${cell.textContent}"`);
      }
    }
    csv.push(rowData.join(","));
  }

  const filename = document.querySelector("#subject").textContent;
  downloadCSV(csv.join("\n"), filename);
}

// Adds event listener for the 'scan' button click.
document.getElementById("scan-button").addEventListener("click", () => {
  scan_and_process_image();
});

// Adds event listener for the 'csv' button click.
document.getElementById("csv-button").addEventListener("click", () => {
  exportStudentsTableToCSV();
});

// Adds event listener for the 'clear' button click.
document.getElementById("clear-button").addEventListener("click", () => {
  const instructions = document.querySelector("#instructions");
  instructions.style.display = "block";
  showDisplayableElementsAfterScan(false);
  clearStudentsTable();
});

// Loads the student data from the JSON file.
let students = [];
fetch("static/students.json")
  .then((response) => response.json())
  .then((data) => {
    students = data;
  })
  .catch((error) => {
    console.error(`Error loading students from json: ${error}.`);
  });
