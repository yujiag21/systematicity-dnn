static void goodG2B()
{
    char * data;
    /* FIX: Initialize data */
    data = "string";
    /* POTENTIAL FLAW: Use data without initializing it */
    printLine(data);
}
