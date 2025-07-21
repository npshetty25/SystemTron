/*
Week 4: Stored Procedures & Optimization
Project: Develop and optimize a billing system using Stored Procedures


 * SQL script to create a simple billing system with Customers, Invoices, and Payments tables.
 * It includes stored procedures and triggers for generating invoices and updating payment status.
 */

DROP DATABASE IF EXISTS BillingSystem;
CREATE DATABASE BillingSystem;
USE BillingSystem;
-- Customers table
CREATE TABLE Customers (
    CustomerID INT PRIMARY KEY AUTO_INCREMENT,
    Name VARCHAR(100),
    Email VARCHAR(100),
    ContactNumber VARCHAR(15)
);

-- Invoices table
CREATE TABLE Invoices (
    InvoiceID INT PRIMARY KEY AUTO_INCREMENT,
    CustomerID INT,
    InvoiceDate DATE,
    Amount DECIMAL(10,2),
    PaymentStatus ENUM('Pending', 'Paid') DEFAULT 'Pending',
    FOREIGN KEY (CustomerID) REFERENCES Customers(CustomerID)
);

-- Payments table
CREATE TABLE Payments (
    PaymentID INT PRIMARY KEY AUTO_INCREMENT,
    InvoiceID INT,
    PaymentDate DATE,
    AmountPaid DECIMAL(10,2),
    FOREIGN KEY (InvoiceID) REFERENCES Invoices(InvoiceID)
);
-- Sample Customers
INSERT INTO Customers (Name, Email, ContactNumber)
VALUES 
('Alice', 'alice@example.com', '9876543210'),
('Bob', 'bob@example.com', '9123456789'),
('Charlie', 'charlie@example.com', '9012345678'),
('Eve', 'eve.a@example.com', '9765432109'),
('Frank', 'frank.m@example.com', '9654321098'),
('Grace','grace.l@example.com', '9543210987');

DELIMITER $$

CREATE PROCEDURE GenerateInvoices()
BEGIN
    DECLARE done INT DEFAULT FALSE;
    DECLARE custID INT;

    DECLARE cust_cursor CURSOR FOR
        SELECT CustomerID FROM Customers;

    DECLARE CONTINUE HANDLER FOR NOT FOUND SET done = TRUE;

    OPEN cust_cursor;

    read_loop: LOOP
        FETCH cust_cursor INTO custID;
        IF done THEN
            LEAVE read_loop;
        END IF;

        INSERT INTO Invoices (CustomerID, InvoiceDate, Amount)
        VALUES (custID, CURDATE(), ROUND(RAND() * 1000, 2));
    END LOOP;

    CLOSE cust_cursor;
END$$

DELIMITER ;
CALL GenerateInvoices();
SELECT * FROM Invoices;


DELIMITER $$

CREATE TRIGGER AfterPaymentUpdateStatus
AFTER INSERT ON Payments
FOR EACH ROW
BEGIN
    UPDATE Invoices
    SET PaymentStatus = 'Paid'
    WHERE InvoiceID = NEW.InvoiceID;
END$$

DELIMITER ;
INSERT INTO Payments (InvoiceID, PaymentDate, AmountPaid)
VALUES (1, CURDATE(), 500.00);

-- Show result
SELECT * FROM Invoices WHERE InvoiceID = 1;

CREATE INDEX idx_customer ON Invoices(CustomerID);
CREATE INDEX idx_invoice ON Payments(InvoiceID);
SET profiling = 1;

SELECT * FROM Invoices WHERE CustomerID = 1;

SHOW PROFILES;
