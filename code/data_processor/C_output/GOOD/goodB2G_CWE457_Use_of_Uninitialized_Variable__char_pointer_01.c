static void goodB2G()
{
    char * data;
    /* POTENTIAL FLAW: Don't initialize data */
    ; /* empty statement needed for some flow variants */
    /* FIX: Ensure data is initialized before use */
    data = "string";
    printLine(data);
}
