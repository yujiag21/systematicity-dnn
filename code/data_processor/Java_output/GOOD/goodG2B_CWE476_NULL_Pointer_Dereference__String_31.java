    private void goodG2B() throws Throwable
    {
        String dataCopy;
        {
            String data;

            /* FIX: hardcode data to non-null */
            data = "This is not null";

            dataCopy = data;
        }
        {
            String data = dataCopy;

            /* POTENTIAL FLAW: null dereference will occur if data is null */
            IO.writeLine("" + data.length());

        }
    }
