/*
Week 3: Advanced Querying & Joins
Project: Analyze employee performance using SQL Joins

 * SQL Code for Employee Performance Management System
 * This code creates tables for employees, departments, and performance reviews,
 * inserts sample data, and provides various queries to analyze employee performance.
 */

-- Employees Table
CREATE TABLE Employees (
    employee_id INTEGER PRIMARY KEY,
    name TEXT NOT NULL,
    dept_id INTEGER,
    hire_date DATE
);

-- Departments Table
CREATE TABLE Departments (
    dept_id INTEGER PRIMARY KEY,
    dept_name TEXT NOT NULL
);

-- PerformanceReviews Table
CREATE TABLE PerformanceReviews (
    review_id INTEGER PRIMARY KEY,
    employee_id INTEGER,
    review_date DATE,
    score INTEGER,
    comments TEXT,
    FOREIGN KEY (employee_id) REFERENCES Employees(employee_id)
);
-- Departments
INSERT INTO Departments (dept_id, dept_name) VALUES
(1, 'Sales'),
(2, 'Engineering'),
(3, 'HR');

-- Employees
INSERT INTO Employees (employee_id, name, dept_id, hire_date) VALUES
(1, 'Alice', 1, '2022-01-10'),
(2, 'Bob', 2, '2021-03-15'),
(3, 'Charlie', 1, '2023-06-01'),
(4, 'Diana', 3, '2020-11-20');

-- PerformanceReviews
INSERT INTO PerformanceReviews (review_id, employee_id, review_date, score, comments) VALUES
(1, 1, '2023-01-15', 85, 'Good'),
(2, 1, '2024-01-15', 90, 'Excellent'),
(3, 2, '2023-02-10', 78, 'Satisfactory'),
(4, 3, '2024-03-12', 88, 'Very Good'),
(5, 4, '2023-12-01', 92, 'Outstanding');





Get all employees with their department names:

SELECT e.employee_id, e.name, d.dept_name
FROM Employees e
INNER JOIN Departments d ON e.dept_id = d.dept_id;


..
List all employees and their latest review score (if any):

SELECT e.name, d.dept_name, pr.score
FROM Employees e
LEFT JOIN Departments d ON e.dept_id = d.dept_id
LEFT JOIN PerformanceReviews pr ON e.employee_id = pr.employee_id;

..

List all departments and employees (including departments with no employees):

SELECT d.dept_name, e.name
FROM Departments d
RIGHT JOIN Employees e ON d.dept_id = e.dept_id;

..

a. Average Performance Score by Department

SELECT d.dept_name, AVG(pr.score) AS avg_score
FROM Departments d
JOIN Employees e ON d.dept_id = e.dept_id
JOIN PerformanceReviews pr ON e.employee_id = pr.employee_id
GROUP BY d.dept_name;

b. Count of Reviews per Employee
SELECT e.name, COUNT(pr.review_id) AS review_count
FROM Employees e
LEFT JOIN PerformanceReviews pr ON e.employee_id = pr.employee_id
GROUP BY e.name;

c. Total Score per Employee
SELECT e.name, SUM(pr.score) AS total_score
FROM Employees e
LEFT JOIN PerformanceReviews pr ON e.employee_id = pr.employee_id
GROUP BY e.name;

..
Subqueries
a. Employees with Above-Average Performance

SELECT e.name
FROM Employees e
JOIN PerformanceReviews pr ON e.employee_id = pr.employee_id
WHERE pr.score > (
    SELECT AVG(score) FROM PerformanceReviews
);

b. Latest Review for Each Employee
SELECT e.name, pr.score, pr.review_date
FROM Employees e
JOIN PerformanceReviews pr ON e.employee_id = pr.employee_id
WHERE pr.review_date = (
    SELECT MAX(review_date)
    FROM PerformanceReviews
    WHERE employee_id = e.employee_id
);

...

 Performance Trend Report Example
 
 SELECT d.dept_name, strftime('%Y', pr.review_date) AS year, AVG(pr.score) AS avg_score
FROM Departments d
JOIN Employees e ON d.dept_id = e.dept_id
JOIN PerformanceReviews pr ON e.employee_id = pr.employee_id
GROUP BY d.dept_name, year
ORDER BY d.dept_name, year;