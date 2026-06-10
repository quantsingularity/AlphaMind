-- AlphaMind Database Initialization
SET GLOBAL sql_mode = 'STRICT_TRANS_TABLES,NO_ZERO_IN_DATE,NO_ZERO_DATE,ERROR_FOR_DIVISION_BY_ZERO,NO_ENGINE_SUBSTITUTION';

CREATE DATABASE IF NOT EXISTS alphamind_db CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;

-- The MySQL Docker entrypoint creates MYSQL_USER before running init scripts;
-- guard with IF NOT EXISTS so the GRANT/REVOKE below never fail on a missing user.
CREATE USER IF NOT EXISTS 'alphamind_user'@'%';

-- Least privilege: strip any global privileges, then grant only on the app DB.
REVOKE ALL PRIVILEGES, GRANT OPTION ON *.* FROM 'alphamind_user'@'%';
GRANT SELECT, INSERT, UPDATE, DELETE, CREATE, INDEX, ALTER ON alphamind_db.* TO 'alphamind_user'@'%';
FLUSH PRIVILEGES;
