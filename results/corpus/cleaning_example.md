# Raw HTML to clean text: one passage of the December 17, 1996 FOMC minutes

Source: https://www.federalreserve.gov/fomc/minutes/19961217.htm (ISO-8859-1, table layout, hard-wrapped paragraphs).

## Raw HTML as served

```html
nd,
                       Senior Vice Presidents, Federal Reserve Banks
                       of San Francisco, Kansas City, Atlanta, and
                       Richmond respectively
<p>		Messrs. Gavin, Kos, and Rosengren, Vice Presidents,
                       Federal Reserve Banks of St. Louis, New York,
                       and Boston respectively
<p>		Mr. Evans, Assistant Vice President, Federal
			Reserve Bank of Chicago

	</td>
	</tr>
</table>

	</td>
	</tr>
</table>

<HR size=+4 width=80% align=left>
<table border=0 cellpadding=5>
	<tr>
	<td width=600>

<P>		By unanimous vote, the minutes of the meeting of the Federal
Open Market Committee held on November 13, 1996, were approved.
<p>		The Manager of the System Open Market Account reported on developments in foreign exchange markets since the meeting on November 13, 1996.  There were no transactions in foreign currencies for System account during this period, and thus no vote was required of the Committee.
<p>		The Manager also reported on developments in domestic
financial markets and on System open market transactions in government
securities and federal agency obligations during the period from
November 13, 1996, through December 16, 1996.  By unanimous vote, the
Committee ratified these transactions.
<p>	  	The Committee members discussed certain changes in the
procedures for conducting domestic open market operations that the
Manager of the System Open Market Account had propo
```

## After scrape_fed.py

```text
ctively
Messrs. Gavin, Kos, and Rosengren, Vice Presidents, Federal Reserve Banks of St. Louis, New York, and Boston respectively
Mr. Evans, Assistant Vice President, Federal Reserve Bank of Chicago

By unanimous vote, the minutes of the meeting of the Federal Open Market Committee held on November 13, 1996, were approved.
The Manager of the System Open Market Account reported on developments in foreign exchange markets since the meeting on November 13, 1996. There were no transactions in foreign currencies for System account during this period, and thus no vote was required of the Committee.
The Manager also reported on developments in domestic financial markets and on System open market transactions in government securities and federal agency obligations during the period from November 13, 1996, through December 16, 1996. By unanimous vote, the Committee ratified these transactions.
Th
```
