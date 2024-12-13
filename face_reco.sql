-- Create the database
CREATE DATABASE face_reco;

-- Use the new database
USE face_reco;

-- Create a 'person_data' table to store person's information
CREATE TABLE person_da (
    id INT AUTO_INCREMENT PRIMARY KEY,      -- Unique ID for the person
    full_name VARCHAR(255) NOT NULL         -- Full name of the person
);

-- Create a 'face_photos' table to store images as LONGBLOB
CREATE TABLE face_pho (
    id INT AUTO_INCREMENT PRIMARY KEY,      -- Unique ID for the image
    person_id INT NOT NULL,                 -- Foreign key linking to person_data table
    image LONGBLOB,                         -- Image stored as binary data
    FOREIGN KEY (person_id) REFERENCES person_da(id) ON DELETE CASCADE -- Ensuring referential integrity
);
select * from face_pho;

-- Create a table to store feature vectors (this was missing)
CREATE TABLE face_features (
    id INT AUTO_INCREMENT PRIMARY KEY,           -- Unique ID for each feature set
    person_id INT NOT NULL,                      -- Foreign key to link features to person
    feature_vector LONGTEXT NOT NULL,            -- Feature vector as a string or serialized numpy array
    FOREIGN KEY (person_id) REFERENCES person_da(id) ON DELETE CASCADE  -- Link to person table
);
select * from face_features;
