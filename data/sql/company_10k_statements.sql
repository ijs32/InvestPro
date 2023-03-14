CREATE TABLE `company_10k_statements` (
  `statement_id` BIGINT(20) NOT NULL AUTO_INCREMENT, 
  `ticker` VARCHAR(10) NOT NULL,  
  `reportDate` DATE NOT NULL,
  `yoy_performance` DECIMAL(8, 2) NOT NULL,
  `predicted_sentiment` DECIMAL(3,2) NOT NULL,
  `10k_statement_text` MEDIUMTEXT NOT NULL,
  PRIMARY KEY (`statement_id`)
)
COLLATE='latin1_swedish_ci'
ENGINE=InnoDB
;