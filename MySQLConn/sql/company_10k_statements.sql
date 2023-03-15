CREATE TABLE `company_10k_statements` (
  `statement_id` BIGINT(20) NOT NULL AUTO_INCREMENT,
  `file_name` VARCHAR(50) NOT NULL,
  `ticker` VARCHAR(10) NOT NULL,
  `exchange` VARCHAR(10) NOT NULL,
  `report_date` DATE NOT NULL,
  `rounded_report_date` DATE NULL,
  `rounded_eoy_date` DATE NULL,
  `yoy_performance` DECIMAL(8,2) DEFAULT NULL,
  `predicted_sentiment` DECIMAL(3,2) DEFAULT NULL,
  `performance_id` BIGINT(20) NOT NULL,
  PRIMARY KEY (`statement_id`)
)
COLLATE='latin1_swedish_ci'
ENGINE=InnoDB
;
