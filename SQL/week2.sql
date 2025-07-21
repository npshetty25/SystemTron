/*
 Week 2: Data Manipulation & Querying
Project: Perform CRUD (Create, Read, Update, Delete) operations on a student database
*/


CREATE TABLE Students (
    student_id INTEGER PRIMARY KEY,
    first_name TEXT NOT NULL,
    last_name TEXT NOT NULL,
    date_of_birth DATE,
    email TEXT UNIQUE
);

CREATE TABLE Courses (
    course_id INTEGER PRIMARY KEY,
    course_name TEXT NOT NULL,
    course_description TEXT
);

CREATE TABLE Enrollments (
    enrollment_id INTEGER PRIMARY KEY,
    student_id INTEGER,
    course_id INTEGER,
    enrollment_date DATE,
    FOREIGN KEY(student_id) REFERENCES Students(student_id),
    FOREIGN KEY(course_id) REFERENCES Courses(course_id)
);
INSERT INTO Students (first_name, last_name, date_of_birth, email)
VALUES 
  ('Aarav', 'Mehra', '2002-08-15', 'aarav.mehra@gmail.com'),
  ('Priya', 'Sharma', '2001-11-22', 'priya.sharma@yahoo.com'),
  ('Rahul', 'Verma', '2003-03-09', 'rahul.verma@hotmail.com'),
  ('Sneha', 'Patel', '2000-12-31', 'sneha.patel@gmail.com'),
  ('Karan', 'Singh', '2002-05-27', 'karan.singh@outlook.com');
INSERT INTO Courses (course_name, course_description)
VALUES
  ('Mathematics', 'An introduction to algebra, calculus, and geometry.'),
  ('English Literature', 'Study of classic and modern English literature.'),
  ('Computer Science', 'Basics of programming, algorithms, and data structures.'),
  ('Physics', 'Fundamentals of mechanics, optics, and thermodynamics.'),
  ('History', 'World history from ancient to modern times.');
INSERT INTO Enrollments (student_id, course_id, enrollment_date)
VALUES
  (1, 3, '2025-01-10'),  -- Aarav in Computer Science
  (2, 1, '2025-01-12'),  -- Priya in Mathematics
  (3, 2, '2025-01-15'),  -- Rahul in English Literature
  (4, 4, '2025-01-18'),  -- Sneha in Physics
  (5, 5, '2025-01-20'),  -- Karan in History
  (1, 1, '2025-01-22'),  -- Aarav in Mathematics
  (2, 3, '2025-01-25');  -- Priya in Computer Science

UPDATE Students
SET email = 'priya.sharma2025@gmail.com'
WHERE first_name = 'Priya' AND last_name = 'Sharma';
UPDATE Students
SET first_name = 'Karanveer'
WHERE student_id = 5;



DELETE FROM Students
WHERE first_name = 'Rahul' AND last_name = 'Verma';
DELETE FROM Students
WHERE student_id = 4;


 --With WHERE
SELECT * FROM Students
WHERE date_of_birth > '2002-01-01';
--With ORDER BY
SELECT * FROM Students
ORDER BY last_name ASC;
--With GROUP BY
SELECT strftime('%Y', date_of_birth) AS birth_year, COUNT(*) AS num_students
FROM Students
GROUP BY birth_year
ORDER BY birth_year;
